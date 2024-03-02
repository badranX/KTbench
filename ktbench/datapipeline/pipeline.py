from torch.utils.data import DataLoader
import torch
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing

from ..pad_utils import splitter, padder, padder_list
from .map_yamld import unfold_mapper, map_yamld, map_yamld_unfold, features_to_tensors, lens2mask
from .middata_manager import download_dataset, gitdownload
from ..train import Trainer

from sklearn.model_selection import KFold
import shutil
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field
import yamld
import os


packagepath = [p for p in Path(__file__).parents if p.name == "ktbench"][0]

from dataclasses import dataclass


REDUCE_PREDICT_KEYS = ['ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_unfold_seq_mask', 'ktbench_label_seq'] 
UNFOLD_KEYS = ['ktbench_exer_unfold_seq', 'ktbench_kc_unfold_seq', 'ktbench_unfold_seq_mask', 'ktbench_label_unfold_seq']
QUESTION_LEVEL_KEYS = ['ktbench_exer_seq', 'ktbench_kc_seq', 'ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_label_seq']
MASKED_UNFOLD_LABELS = ['ktbench_masked_label_unfold_seq']
TEACHER_MASKS= ['ktbench_teacher_unfold_seq_mask']

SEQUENCE_AXIS = {'ktbench_exer_seq_mask': -1,
                  'ktbench_kc_seq_mask': -2,
                  'ktbench_unfold_seq_mask': -1,
                   'ktbench_label_seq': -1,
                'ktbench_exer_unfold_seq': -1,
                'ktbench_kc_unfold_seq': -1,
                'ktbench_unfold_seq_mask': -1,
                'ktbench_label_unfold_seq': -1,
                'ktbench_exer_seq': -1,
                'ktbench_kc_seq':-2}

def rename_columns(ds, featuremap):
    features = ds.features.values()
    for old, new in featuremap.items():
        if old == new:
            continue
        ds = ds.rename_column(old, new)
    return ds


class Pipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.dataset2model_feature_map = self.cfg.model_cls.MODEL_FEATURE_MAP
        self.window_size = cfg.window_size
        self.dataset_name = cfg.dataset_name
        self.is_unfold = cfg.is_unfold

        self.process_device = 'cpu' 
        if hasattr(cfg, 'device'):
            self.device = cfg.device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.cfg.device = self.device


        if hasattr(cfg, 'extra_features'):
            self.extra_features = cfg.extra_features
        else:
            self.extra_features = dict()


        if hasattr(cfg, 'is_unfold_fixed_window'):
            self.is_unfold_fixed_window = cfg.is_unfold_fixed_window
        else:
            self.is_unfold_fixed_window = False

        if hasattr(cfg, 'dataset_path'):
            self.dataset_dir = Path(cfg.dataset_path).parent
            self.yaml_dataset_path = Path(cfg.dataset_path)
        else:
            self.dataset_dir = Path.cwd()
            self.yaml_dataset_path = self.dataset_dir / (self.dataset_name + '.yaml')

        
        self.splits = [0.6, 0.5] if not hasattr(cfg, 'splits') else cfg.splits
        self.kfolds = 1 if not hasattr(cfg, 'kfold') else cfg.kfold
        self.multi2one_kcs = False if not hasattr(cfg, 'multi2one_kcs') else cfg.multi2one_kcs

        self.add_hide_label = False if not hasattr(cfg, 'add_hide_label') else cfg.add_hide_label
        self.add_teacher_mask = False if not hasattr(cfg, 'add_teacher_mask') else cfg.add_teacher_mask
        self.init_tgt_features()

        
    def init_tgt_features(self):
        extra_features = list(self.extra_features.keys())

        if self.is_unfold:
            tgt_features = UNFOLD_KEYS + extra_features
            if self.add_hide_label:
                tgt_features += MASKED_UNFOLD_LABELS
            if self.add_teacher_mask:
                tgt_features += TEACHER_MASKS
            if self.cfg.eval_method == Trainer.EVAL_UNFOLD_REDUCE:
                eval_tgt_features = list(set(tgt_features + REDUCE_PREDICT_KEYS))
            else:
                eval_tgt_features = tgt_features
            
            if self.is_unfold_fixed_window:
                #Mostly it is done this way in other works
                tgt_features = extra_features + [x for x in tgt_features if 'unfold' in x or x == 'stu_id']
                eval_tgt_features = extra_features + [x for x in eval_tgt_features if 'unfold' in x or x == 'stu_id']

        else:
            tgt_features = QUESTION_LEVEL_KEYS + extra_features
            eval_tgt_features = tgt_features
        self.tgt_features = tgt_features
        self.eval_tgt_features = eval_tgt_features


    def generate_hg_dataset(self):
        ds = self.yamld2dataset()
        #ds = ds.select_columns(all_features)
        ds = ds.with_format("torch", device= self.device)
        return ds 

    
    def split_dataset(self, ds):
        l_train_ds  = []
        l_valid_ds  = []

        if self.kfolds == 1:

            train_ds, extra_ds = ds.train_test_split(train_size=self.splits[0], shuffle=True, seed=32).values()
            test_ds, valid_ds = extra_ds.train_test_split(train_size=self.splits[1], shuffle=True, seed=32).values()
            train_ds = train_ds.select_columns(self.tgt_features)
            valid_ds = valid_ds.select_columns(self.eval_tgt_features)
            test_ds = test_ds.select_columns(self.eval_tgt_features)
            train_ds = rename_columns(train_ds, self.cfg.dataset2model_feature_map)
            test_ds = rename_columns(test_ds, self.cfg.dataset2model_feature_map)
            valid_ds = rename_columns(valid_ds, self.cfg.dataset2model_feature_map)
            l_train_ds.append(train_ds)
            l_valid_ds.append(valid_ds)

        else:

            rest_ds, test_ds = ds.train_test_split(train_size=(1-self.splits[0])*self.splits[1], shuffle=True, seed=32).values()
            test_ds = test_ds.select_columns(self.eval_tgt_features)
            test_ds = rename_columns(test_ds, self.cfg.dataset2model_feature_map)
            folds = KFold(n_splits=self.kfolds, shuffle=True, random_state=2)
            splits = folds.split(np.zeros(rest_ds.num_rows))
            for train_idxs, valid_idxs in splits:
                train_ds = rest_ds.select(train_idxs)
                valid_ds = rest_ds.select(valid_idxs)
                valid_ds = valid_ds.select_columns(self.eval_tgt_features)
                train_ds = train_ds.select_columns(self.tgt_features)

                train_ds = rename_columns(train_ds, self.cfg.dataset2model_feature_map)
                valid_ds = rename_columns(valid_ds, self.cfg.dataset2model_feature_map)
                l_train_ds.append(train_ds)
                l_valid_ds.append(valid_ds)

        ret = {
            'train_ds' : l_train_ds,
            'valid_ds' : l_valid_ds,
            'test_ds' : test_ds,
        }

        self.cfg.__dict__.update(ret)
        ret.update(ret)
        return SimpleNamespace(**ret)
        

    @staticmethod
    def mapper(x, window_size):
        exer_seq, lens, idx = padder(
            x['exer_id'],  out_maxlen=window_size, use_out_maxlen=True
        )
        label_seq, _, _ = padder(
            x['label'],
       out_maxlen=window_size, use_out_maxlen=True
        )
        if idx:
            stu_id = x['stu_id'][idx]
        else:
            stu_id = x['stu_id']

        ret_dict = {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'lens_seq': lens 
        }
        return ret_dict

    def is_middata_ready(self):
        return self.yaml_dataset_path.exists()
    
    def get_meta(self):
        yamld_file = self.yaml_dataset_path
        meta = yamld.read_metadata(yamld_file)
        return meta

    def process_dataset(self, ds, meta=None):
        if not meta:
            yamld_file = self.yaml_dataset_path
            meta = yamld.read_metadata(yamld_file)
        #TODO
        ds = ds.with_format("torch", device= self.process_device)
        map1 = lambda x: self.mapper(x, window_size=self.window_size)
        ds = ds.map(map1, batched=True, remove_columns=ds.column_names)

        problem2KCs = meta['problem2KCs']
        if type(problem2KCs) is dict:
            problem2KCs = {int(k): v for k, v in problem2KCs.items()}
            d = zip(*sorted(problem2KCs.items(), key=lambda x: x[0]))
            keys = next(d)
            assert tuple(range(len(keys))) == keys
            problem2KCs = list(next(d))

        if self.multi2one_kcs:
            problem2KCs = list(map(tuple, problem2KCs))
            problem2KCs, _ = pd.factorize(pd.Series(problem2KCs))

            meta['n_kc'] = max(problem2KCs) + 1
            problem2KCs = list(map(lambda x: [x], problem2KCs))
            meta['kc_seq_padding'] = problem2KCs
            meta['kc_seq_lens'] = [[1]]*len(problem2KCs)
            max_kcs_per_exer = 1
            meta['max_kcs_per_exer'] = max_kcs_per_exer
            meta['kc_seq_mask'] = lens2mask(kc_seq_lens, max_kcs_per_exer)

            self.cfg.n_kc=meta['n_kc']
        else:
            problem2KCs, kc_seq_lens, idx = padder_list(problem2KCs, dtype=int, to_tensor=True)
            #TODO middata processing
            meta['kc_seq_padding'] = problem2KCs
            meta['kc_seq_lens'] = kc_seq_lens
            max_kcs_per_exer = kc_seq_lens.max()
            meta['max_kcs_per_exer'] = max_kcs_per_exer
            meta['kc_seq_mask'] = lens2mask(kc_seq_lens, max_kcs_per_exer)

        meta['max_exer_window_size'] = self.window_size
        features_to_tensors(meta)
        if self.is_unfold:
            map2 = lambda x: map_yamld_unfold(x, meta, is_hide_label=self.add_hide_label or self.add_teacher_mask)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)
            if self.is_unfold_fixed_window:
                map3 = lambda x: unfold_mapper(x, window_size=self.window_size)
                ds = ds.map(map3, batched=True, remove_columns=ds.column_names)
        else:
            map2 = lambda x: map_yamld(x, meta)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)

        rename = dict(zip(ds.column_names, map(lambda x: 'ktbench_' + x, ds.column_names)))
        ds = ds.rename_columns(rename)
        #ds = ds.select_columns(all_features)
        return ds
        

    def yamld2dataset(self):
        yamld_file = self.yaml_dataset_path
        meta = yamld.read_metadata(yamld_file)
        self.meta = meta
        readgen = yamld.read_generator(yamld_file)

        ds = Dataset.from_generator(readgen, cache_dir="./.cache_dir")
        #TODO
        ds = ds.with_format("torch", device= self.process_device)

        map1 = lambda x: self.mapper(x, window_size=self.window_size)
        ds = ds.map(map1, batched=True, remove_columns=ds.column_names)

        problem2KCs = meta['problem2KCs']
        if type(problem2KCs) is dict:
            problem2KCs = {int(k): v for k, v in problem2KCs.items()}
            d = zip(*sorted(problem2KCs.items(), key=lambda x: x[0]))
            keys = next(d)
            assert tuple(range(len(keys))) == keys
            problem2KCs = list(next(d))

        if self.multi2one_kcs:
            problem2KCs = list(map(tuple, problem2KCs))
            problem2KCs, _ = pd.factorize(problem2KCs)

            meta['n_kc'] = max(problem2KCs) + 1
            problem2KCs = list(map(lambda x: [x], problem2KCs))
            meta['kc_seq_padding'] = torch.tensor(problem2KCs, dtype=int)
            kc_seq_lens = torch.tensor([1]*len(problem2KCs), dtype=int)
            meta['kc_seq_lens'] = kc_seq_lens
            max_kcs_per_exer = 1
            meta['max_kcs_per_exer'] = max_kcs_per_exer
            meta['kc_seq_mask'] = lens2mask(kc_seq_lens, max_kcs_per_exer)

            self.cfg.n_kc=meta['n_kc']
        else:
            problem2KCs, kc_seq_lens, idx = padder_list(problem2KCs, dtype=int, to_tensor=True)
            #TODO middata processing
            meta['kc_seq_padding'] = problem2KCs
            meta['kc_seq_lens'] = kc_seq_lens
            max_kcs_per_exer = kc_seq_lens.max()
            meta['max_kcs_per_exer'] = max_kcs_per_exer
            meta['kc_seq_mask'] = lens2mask(kc_seq_lens, max_kcs_per_exer)

        meta['max_exer_window_size'] = self.window_size
        features_to_tensors(meta)
        if self.is_unfold:
            map2 = lambda x: map_yamld_unfold(x, meta, is_hide_label=self.add_hide_label or self.add_teacher_mask)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)
            if self.is_unfold_fixed_window:
                map3 = lambda x: unfold_mapper(x, window_size=self.window_size)
                ds = ds.map(map3, batched=True, remove_columns=ds.column_names)
        else:
            map2 = lambda x: map_yamld(x, meta)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)

        rename = dict(zip(ds.column_names, map(lambda x:'ktbench_' + x, ds.column_names)))
        ds = ds.rename_columns(rename)
        #ds = ds.select_columns(all_features)
        return ds
        

    def start(self, do_split=True, gen=None, from_middata=False):
        extra_features = list(self.extra_features.keys())

        if not self.is_middata_ready():
            shutil.rmtree(self.yaml_dataset_path, ignore_errors=True)
            #self.yaml_dataset_path = download_dataset(self.dataset_name, self.dataset_dir)
            self.yaml_dataset_path = gitdownload(self.dataset_name, self.dataset_dir)

        meta = yamld.read_metadata(self.yaml_dataset_path)

        self.cfg.n_exer=meta['n_exer']
        self.cfg.n_kc=meta['n_kc']
        if 'n_stu' in meta:
            self.cfg.n_stu = meta['n_stu']
        else:
            self.cfg.n_stu= -1
        if not self.window_size:
            print('[WARNING] window_size was not provided, default to max')
            self.cfg.window_size = meta['max_exer_window_size'] 
        self.cfg.dataset_name=self.dataset_name
        self.cfg.max_window_size=self.window_size

        #allcfg = Config(device=self.device, dataset_info=dsinfo)
        ret = {
            'meta': meta,
        }
        
        ds = self.generate_hg_dataset()
        
        if not do_split:
            return ds 
        else:
            return self.split_dataset(ds)


if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class Cfg:
        dataset_name = "assist2009"
        window_size = 100
        yaml_middata = './test.yaml'

    pipes = Pipeline(Cfg())
    dd = pipes.start()
    print(dd)
    #df.to_csv("deleteme.csv")
    #print(extras)