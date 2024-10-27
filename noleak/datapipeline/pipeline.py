from torch.utils.data import DataLoader
import copy
import torch
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold

from ..pad_utils import splitter, padder, padder_list
from .map_yamlx import unfold_mapper, map_yamlx, map_yamlx_unfold, features_to_tensors, lens2mask
from .map_yamlx import map_allinone_batch, map_allinone_before_batch

from .middata_manager import download_dataset, gitdownload

from sklearn.model_selection import KFold
import shutil
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field
from .. import yamlx
import os
from dataclasses import dataclass


KTBENCH_DATASETS_FOLDER = ".datasets_ktbench"
SEED = 42
REDUCE_PREDICT_KEYS = ['ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_unfold_seq_mask', 'ktbench_label_seq'] 
UNFOLD_KEYS = ['ktbench_exer_unfold_seq', 'ktbench_kc_unfold_seq', 'ktbench_unfold_seq_mask', 'ktbench_label_unfold_seq']
QUESTION_LEVEL_KEYS = ['ktbench_exer_seq', 'ktbench_kc_seq', 'ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_label_seq']
MASKED_UNFOLD_LABELS = ['ktbench_masked_label_unfold_seq']
TEACHER_MASKS= ['ktbench_teacher_unfold_seq_mask']
ATTENTION_MASKS = ['ktbench_attention_mask', 'ktbench_kc_seq_mask']

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


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class Pipeline():
    EVAL_QUESTION_LEVEL, \
    EVAL_UNFOLD_KC_LEVEL, \
    EVAL_UNFOLD_REDUCE, \
    EVAL_UNFOLD_STEP, \
    *_ = range(10)
    
    TEST_LIKE_EVAL, \
    TEST_STEP, \
    *_ = range(10)
    def __init__(self, cfg):
        self.cfg = cfg

        self.seed = getattr(self.cfg, 'seed', SEED)
        seed_everything(self.seed)
        self.cfg.dataset2model_feature_map = getattr(self.cfg.model_cls, 'MODEL_FEATURE_MAP', {})

        self.window_size = cfg.window_size
        self.dataset_name = cfg.dataset_name
        self.is_unfold = cfg.is_unfold

        self.process_device = 'cpu' 
        if hasattr(cfg, 'device'):
            self.device = cfg.device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.cfg.device = self.device

        self.extra_features = getattr(cfg, 'extra_features', dict())
        self.is_attention = getattr(cfg, 'is_attention', False)

        self.is_unfold_fixed_window = getattr(cfg, 'is_unfold_fixed_window', False)

        if hasattr(cfg, 'dataset_path'):
            self.dataset_dir = Path(cfg.dataset_path).parent
            self.yaml_dataset_path = Path(cfg.dataset_path)
        else:
            self.dataset_dir  = Path.cwd() / KTBENCH_DATASETS_FOLDER
            if not self.dataset_dir.exists():
                self.dataset_dir.mkdir()
            self.yaml_dataset_path = self.dataset_dir / (self.dataset_name + '.yaml')

        
        self.splits = getattr(cfg, 'splits', [0.2, 0.8])
        self.kfolds = getattr(cfg, 'kfold', 1)
        self.multi2one_kcs = getattr(cfg, 'multi2one_kcs', False)

        self.add_mask_label = getattr(cfg, 'add_mask_label', False)
        self.add_teacher_mask = getattr(cfg, 'add_teacher_mask', False)
        self.cfg.all_in_one = getattr(cfg, 'all_in_one', False)
        self.cfg.is_test_all_in_one = getattr(cfg, 'is_test_all_in_one', False)
        self.init_tgt_features()

        
    def init_tgt_features(self):
        extra_features = list(self.extra_features.keys())

        if self.is_unfold:
            tgt_features = UNFOLD_KEYS + extra_features
            if self.add_mask_label:
                tgt_features += MASKED_UNFOLD_LABELS
            if self.add_teacher_mask:
                tgt_features += TEACHER_MASKS
            if self.is_attention:
                tgt_features += ATTENTION_MASKS
            if self.cfg.eval_method == self.EVAL_UNFOLD_REDUCE:
                eval_tgt_features = list(set(tgt_features + REDUCE_PREDICT_KEYS))
            else:
                eval_tgt_features = tgt_features
            
            if self.is_unfold_fixed_window:
                assert self.cfg.eval_method != self.EVAL_UNFOLD_REDUCE
                #Mostly it is done this way in other works
                tgt_features = extra_features + [x for x in tgt_features if 'unfold' in x or x == 'stu_id']
                eval_tgt_features = extra_features + [x for x in eval_tgt_features if 'unfold' in x or x == 'stu_id']

        else:
            tgt_features = QUESTION_LEVEL_KEYS + extra_features
            eval_tgt_features = list(set(tgt_features + ['ktbench_stu_id']))
        self.tgt_features = list(set(tgt_features + ['ktbench_stu_id']))
        self.eval_tgt_features = list(set(eval_tgt_features + ['ktbench_stu_id']))

        self.cfg.eval_tgt_features = eval_tgt_features


    def generate_hg_dataset(self):
        ds = self.yamlx2dataset()
        #ds = ds.select_columns(all_features)
        ds = ds.with_format("torch", device= self.device)
        return ds 

    @staticmethod
    def prepare_all_in_one(test_ds, cfg):
        new_column = range(len(test_ds))
        test_ds= test_ds.add_column("ktbench_idx", new_column)
        print('[INFO] start all_in_one test dataset processing...')
        print('[INFO] test len: ', len(test_ds))
        add_mask_label = getattr(cfg,'add_mask_label', False)
        add_teacher_mask = getattr(cfg,'add_teacher_mask', False)
        map = lambda x: map_allinone_before_batch(x, tgt_features= cfg.eval_tgt_features, is_hide_label=add_mask_label or add_teacher_mask)
        test_ds = test_ds.map(map, batched=False, remove_columns=test_ds.column_names)
        #test_ds = test_ds.remove_columns("ktbench_idx") 
        test_ds = test_ds.map(map_allinone_batch, batched=True, batch_size=1, remove_columns=test_ds.column_names)
        test_ds= test_ds.with_format("torch", device= cfg.device)
        return test_ds
    
    def split_data_by_indices(self, df, test_ratio, n_splits=5):
        # Step 1: Group by unique values and get original indices for each group
        print("indexing dataframe columnds: ", df.columns)
        grouped = df.groupby(df.columns[0]).apply(lambda x: x.index.tolist()).reset_index(name='indices')

        # Step 2: Split into train/test sets
        train_data, test_data = train_test_split(grouped, test_size=test_ratio, random_state=self.seed)
        
        # Extract indices for train and test sets
        #train_indices = [idx for sublist in train_data['indices'] for idx in sublist]
        test_indices = [idx for sublist in test_data['indices'] for idx in sublist]
        #print('indices for test dataset: ', len(test_indices))
        #print('indices for test dataset: ', test_data.columns)

        # Step 3: Split train data into k-folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        train_folds = []
        
        for train_idx, val_idx in kf.split(train_data):
            train_fold_indices = [idx for sublist in train_data.iloc[train_idx]['indices'] for idx in sublist]
            val_fold_indices = [idx for sublist in train_data.iloc[val_idx]['indices'] for idx in sublist]
            
             # Collect indices for the current fold
            train_fold_indices = [idx for sublist in train_data.iloc[train_idx]['indices'] for idx in sublist]
            val_fold_indices = [idx for sublist in train_data.iloc[val_idx]['indices'] for idx in sublist]
            train_folds.append((train_fold_indices, val_fold_indices))

        return train_folds, test_indices
    


    def split_dataset(self, ds):
        l_train_ds  = []
        l_valid_ds  = []

        ds = ds.shuffle(seed=self.seed)
        print('[INFO] total dataset lenght: ', len(ds))

        test_per = self.splits[0]
        
        train_valid_split_per = self.splits[1]
        if False and self.kfolds == 1:
            #TODO this is not used in the paper, split by students instead
            test_ds, extra_ds= ds.train_test_split(train_size=test_per, shuffle=True, seed=self.seed).values()
            train_ds, valid_ds = extra_ds.train_test_split(train_size=train_valid_split_per, shuffle=True, seed=self.seed).values()
            train_ds = train_ds.select_columns(self.tgt_features)
            valid_ds = valid_ds.select_columns(self.eval_tgt_features)
            if False and self.cfg.all_in_one:
                test_ds = self.prepare_all_in_one(test_ds, self.cfg)

             
            if getattr(self.cfg, 'is_test_all_in_one', False):
                test_test_ds = copy.deepcopy(test_ds)
                test_test_ds = rename_columns(test_test_ds, self.cfg.dataset2model_feature_map)
                self.cfg.test_test_ds = test_test_ds

            if not self.cfg.all_in_one:
                test_ds = test_ds.select_columns(self.eval_tgt_features)
                test_ds = rename_columns(test_ds, self.cfg.dataset2model_feature_map)

            train_ds = rename_columns(train_ds, self.cfg.dataset2model_feature_map)
            valid_ds = rename_columns(valid_ds, self.cfg.dataset2model_feature_map)
            l_train_ds.append(train_ds)
            l_valid_ds.append(valid_ds)

        else:
            df = ds.select_columns(['ktbench_stu_id']).to_pandas()
            test_per = self.splits[0]
            train_folds, test_inds = self.split_data_by_indices(df, test_ratio=test_per, n_splits=self.kfolds)
            test_ds = ds.select(test_inds)


            if False and self.cfg.all_in_one:
                test_ds = self.prepare_all_in_one(test_ds, self.cfg)

            if not self.cfg.all_in_one:
                test_ds = test_ds.select_columns(self.eval_tgt_features)
                test_ds = rename_columns(test_ds, self.cfg.dataset2model_feature_map)

            for train_idxs, valid_idxs in train_folds:
                train_ds = ds.select(train_idxs)
                valid_ds = ds.select(valid_idxs)
                valid_ds = valid_ds.select_columns(self.eval_tgt_features)
                train_ds = train_ds.select_columns(self.tgt_features)

                train_ds = rename_columns(train_ds, self.cfg.dataset2model_feature_map)
                valid_ds = rename_columns(valid_ds, self.cfg.dataset2model_feature_map)
                l_train_ds.append(train_ds)
                l_valid_ds.append(valid_ds)

        stu_id = 'ktbench_stu_id'
        l_train_ds = [dds.remove_columns([stu_id]) if stu_id in dds.column_names else dds for dds in l_train_ds]
        l_valid_ds = [dds.remove_columns([stu_id]) if stu_id in dds.column_names else dds for dds in l_valid_ds]

        if stu_id in test_ds.column_names:
            test_ds = test_ds.remove_columns(['ktbench_stu_id'])

        ret = {
            'train_ds' : l_train_ds,
            'valid_ds' : l_valid_ds,
            'test_ds' : test_ds,
        }

        self.cfg.__dict__.update(ret)
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
        yamlx_file = self.yaml_dataset_path
        meta = yamlx.read_metadata(yamlx_file)
        return meta

    def yamlx2dataset(self):
        yamlx_file = self.yaml_dataset_path
        meta = yamlx.read_metadata(yamlx_file)
        self.meta = meta
        readgen = yamlx.read_generator(yamlx_file)

        ds = Dataset.from_generator(readgen, cache_dir="./.cache_dir")
        return self.process_dataset(ds, meta)

    def process_dataset(self, ds, meta):
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
            map2 = lambda x: map_yamlx_unfold(x, meta, is_attention=self.is_attention, is_mask_label=self.add_mask_label, is_teacher_mask= self.add_teacher_mask)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)
            if self.is_unfold_fixed_window:
                map3 = lambda x: unfold_mapper(x, window_size=self.window_size)
                ds = ds.map(map3, batched=True, remove_columns=ds.column_names)
        else:
            map2 = lambda x: map_yamlx(x, meta)
            ds = ds.map(map2, batched=False, remove_columns=ds.column_names)

        rename = dict(zip(ds.column_names, map(lambda x:'ktbench_' + x, ds.column_names)))
        ds = ds.rename_columns(rename)
        #ds = ds.select_columns(all_features)
        return ds
        

    def start(self, do_split=True):
        extra_features = list(self.extra_features.keys())

        if not self.is_middata_ready():
            shutil.rmtree(self.yaml_dataset_path, ignore_errors=True)
            #self.yaml_dataset_path = download_dataset(self.dataset_name, self.dataset_dir)
            self.yaml_dataset_path = gitdownload(self.dataset_name, self.dataset_dir)

        meta = yamlx.read_metadata(self.yaml_dataset_path)

        self.cfg.n_exer=meta['n_exer']
        self.cfg.n_kc=meta['n_kc']
        self.cfg.avg_kc_per_exer = meta['avg_kc_per_exer']
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
