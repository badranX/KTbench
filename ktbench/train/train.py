from torch import autograd
import torch.nn.functional as F
import random

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader#, load_from_disk
import datetime
from ..trainlogs import LogsHandler
import os
from dataclasses import dataclass
import torch
from sklearn import metrics
from tqdm.auto import tqdm
import yamld
from pathlib import Path
import os
import numpy as np
import pandas as pd

DATE_FORMAT = "M%MS%SH%H_d%d_m%m_y%Y"
SEED = 82


class Collate:
    def __init__(self, seqs=None, is_attention=False, seqofseq=None):
        self.seqs = seqs
        self.is_attention = is_attention

    def attention_pad_collate(self, batch):

        all_seqs = batch[0].keys()
        zlens = zip(*map(lambda x: x.values(), batch))
        lens = {'lens_' + str(k)  : vlen for k, vlen in zip(batch[0].keys(), zlens) if k in self.seqs}
        shape0 = len(batch)
        #shape1s = map(len, tmp)
        ak = 'ktbench_attention_mask'
        max_seq = max(map(lambda x: len(x[ak]), batch))
        f = lambda x: max_seq - x[ak].shape[0]
        padseq = map(lambda x: F.pad(x, (0, f(x)), 'constant', 0), batch)
        padseq = map(lambda x: F.pad(x.T, (0, f(x)).T, 'constant', 0), padseq)
        #attention_seq = map(lambda x: list(x[ak]), batch)
        z = zip(*map(lambda x: x.values(), batch))
        batch = {k: pad_sequence(v, batch_first=True, padding_value=0)
                       if k in self.seqs else torch.stack(v)
               for k, v in zip(batch[0].keys(), z) if k != ak}

        batch[ak] = padseq
        batch.update(lens)        
        return batch

    def pad_collate(self, batch):
        if self.is_attention:
            return self.attention_pad_collate(batch)

        all_seqs = batch[0].keys()

        zlens = zip(*map(lambda x: x.values(), batch))
        lens = {'lens_' + str(k)  : vlen for k, vlen in zip(batch[0].keys(), zlens) if k in self.seqs}

        z = zip(*map(lambda x: x.values(), batch))
        batch = {k: pad_sequence(v, batch_first=True, padding_value=0)
                       if k in self.seqs else torch.stack(v)
               for k, v in zip(batch[0].keys(), z)}
        batch.update(lens)        
        return batch

@dataclass
class Point:
    dataset=3
    batch_size=4

#todo caching
#Dataset.from_dict({"data": data}).save_to_disk("my_dataset")


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def compute_metrics(all_target, all_pred):
    return {
        'auc': compute_auc(all_target, all_pred),
        'acc': compute_accuracy(all_target, all_pred)
    }


class Trainer():

    EVAL_QUESTION_LEVEL, \
    EVAL_UNFOLD_KC_LEVEL, \
    EVAL_UNFOLD_REDUCE, \
    EVAL_UNFOLD_STEP, \
    *_ = range(10)
    
    TEST_LIKE_EVAL, \
    TEST_STEP, \
    *_ = range(10)

    def __init__(self, traincfg, cfg, hyper_param=None):
        self.inference_methods = {
            self.EVAL_QUESTION_LEVEL: self.question_eval,
            self.EVAL_UNFOLD_REDUCE: self.reduce_eval,
            self.EVAL_UNFOLD_KC_LEVEL: self.kc_eval,
        }
        self.hyper_param = hyper_param
        self.cfg = cfg
        self.traincfg = traincfg
        self.traincfg.betas = (0.9, 0.999)
        self.traincfg.eps = 1e-8
        
        self.is_unfold = cfg.is_unfold
        self.logs = LogsHandler(cfg)
        cfg.logs = self.logs
        self.cfg = cfg
        self.device =cfg.device
        self.is_padded = getattr(traincfg, 'is_padded', False)
        self.kfolds = getattr(cfg, 'kfold', 1)
        self.is_attention= getattr(cfg, 'is_attention', False)
        self.n_stop_check = getattr(traincfg,'n_stop_check', 10)
        self.seed = getattr(self.cfg, 'seed', SEED)
        if hasattr(cfg, 'eval_method'):
            eval_method = cfg.eval_method
        else:
            if not self.is_unfold:
                eval_method = self.EVAL_QUESTION_LEVEL
            else:
                eval_method = self.EVAL_UNFOLD_REDUCE
        try:
            self.eval_method = self.inference_methods[eval_method]
        except KeyError as e:
            Exception('Unsuperted inference method : ' + e.message)

        self.n_epoch = self.traincfg.n_epoch

        
    def init_model(self):
        if self.hyper_param:
            self.model = self.cfg.model_cls(self.cfg, self.hyper_param).to(self.cfg.device)
        else:
            self.model = self.cfg.model_cls(self.cfg).to(self.cfg.device)
        self.traincfg.model  = self.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.traincfg.lr, betas=self.traincfg.betas, eps=self.traincfg.eps)
        return self.model

    def init_dataloader(self, k):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        k = k - 1 #get index
        seqs = [v for k, v in self.cfg.dataset2model_feature_map.items() if 'unfold' in k]
        if hasattr(self.cfg, 'extra_features'):
            extra = [k for k, v in self.cfg.extra_features.items() if 'unfold' in v]
        else:
            extra = []
        #TODO add split train, valid, test extras
        seqs_train = extra + seqs + [v for v in self.cfg.train_ds[k].column_names if 'unfold' in v]
        seqs_valid = extra + seqs + [v for v in self.cfg.valid_ds[k].column_names if 'unfold' in v]
        seqs_test = extra + seqs + [v for v in self.cfg.test_ds.column_names if 'unfold' in v]
        clt_train = Collate(seqs = seqs_train, is_attention=self.is_attention)
        clt_test = Collate(seqs = seqs_test, is_attention=self.is_attention)
        clt_valid = Collate(seqs = seqs_valid, is_attention=self.is_attention)
        self.train_dataloader = DataLoader(self.cfg.train_ds[k], worker_init_fn=seed_worker, shuffle=True, batch_size=self.traincfg.batch_size, collate_fn=clt_train.pad_collate)
        self.valid_dataloader = DataLoader(self.cfg.valid_ds[k], shuffle=False, batch_size=self.traincfg.eval_batch_size, collate_fn=clt_valid.pad_collate)
        self.test_dataloader = DataLoader(self.cfg.test_ds, shuffle=False, batch_size=self.traincfg.eval_batch_size, collate_fn= clt_test.pad_collate)


    def question_eval(self, y_pd, idxslice, dataset2model_feature_map, **kwargs):
        key_exer_seq_mask = dataset2model_feature_map.get(*2*('ktbench_exer_seq_mask',)) 
        mask = kwargs[key_exer_seq_mask]

        assert (not idxslice.start or idxslice.start <=1) and (not idxslice.stop or idxslice.stop >= -1)
        start = 1 if idxslice.start is None else None
        stop = -1 if idxslice.stop is None else None
        y_pd = y_pd[...,start:stop]
        y_pd = y_pd[mask[...,1:-1] == 1]

        key_ktbench_label_seq = dataset2model_feature_map.get(*2*('ktbench_label_seq',))
        y_gt = kwargs[key_ktbench_label_seq][:, 1:-1]  #remove 1st question

        y_gt = y_gt[mask[:,1:-1]==1]
        
        return {
            'predict': y_pd,
            'target': y_gt,
            'len': len(y_pd)
        }

    def kc_eval(self, y_pd, idxslice, dataset2model_feature_map, **kwargs):
        unfold_seq_mask = dataset2model_feature_map.get(*2*('ktbench_unfold_seq_mask',)) 
        key_ktbench_label_unfold_seq = dataset2model_feature_map.get(*2*('ktbench_label_unfold_seq',))
        mask = kwargs[unfold_seq_mask]

        #assert (not idxslice.start or idxslice.start <=1) and (not idxslice.stop or idxslice.stop >= -1)
        #start = 1 if idxslice.start is None else None
        #stop = -1 if idxslice.stop is None else None
        #y_pd = y_pd[...,start:stop]
        #y_pd = y_pd[...,idxslice]
        y_pd = y_pd[mask[...,idxslice] == 1]

        y_gt = kwargs[key_ktbench_label_unfold_seq][:, idxslice]

        y_gt = y_gt[mask[:, idxslice]==1]
        
        return {
            'predict': y_pd,
            'target': y_gt,
            'len': len(y_pd)
        }
        
    def step_unfold_eval(self, y_pd, idx, dataset2model_feature_map, **kwargs):
        unfold_seq_mask = dataset2model_feature_map.get(*2*('ktbench_unfold_seq_mask',)) 
        mask = kwargs[unfold_seq_mask]

        y_pd = y_pd[...,-1]
        y_pd = y_pd[mask[...,idx] == 1]

        y_gt = y_gt[mask[...,idx]==1]
        
        return {
            'predict': y_pd,
            'target': y_gt,
            'len': len(y_pd)
        }
        
    def reduce_eval(self, y_pd, idxslice, dataset2model_feature_map, **kwargs):
        key_exer_seq_mask = dataset2model_feature_map.get(*2*('ktbench_exer_seq_mask',)) 
        key_kc_seq_mask = dataset2model_feature_map.get(*2*('ktbench_kc_seq_mask',))
        key_unfold_seq_mask = dataset2model_feature_map.get(*2*('ktbench_unfold_seq_mask',))
        key_ktbench_label_seq = dataset2model_feature_map.get(*2*('ktbench_label_seq',))

        mask = kwargs[key_exer_seq_mask]
        kc_seq_mask = kwargs[key_kc_seq_mask]
        unfold_seq_mask = kwargs[key_unfold_seq_mask]

        #prepare constant window-length
        tmpcat = []
        if idxslice.start:
            prepend= -1*torch.ones(*unfold_seq_mask.shape[:-1], idxslice.start)
            tmpcat.append(prepend)
        tmpcat.append(y_pd)
        if idxslice.stop:
            append= -1*torch.ones(*unfold_seq_mask.shape[:-1], idxslice.stop)
            tmpcat.append(append)
        if len(tmpcat) > 1:
            y_pd = torch.cat(tmpcat, dim=-1)
        #end prepare constant window length

        lens = kc_seq_mask[:,1:-1].sum(-1)[mask[:,1:-1]==1]
        #tmp = torch.zeros(*kc_seq_mask.shape, dtype=y_pd.dtype).to(self.device)
        tmp = kc_seq_mask.float()
        #todo make sure masked exersies, is treated as exer 0 and mapped in kc_seq_mask
        tmp[kc_seq_mask==1] = y_pd[unfold_seq_mask == 1]
        #TODO adjust this
        tmp = tmp[:,1:-1]  #remove 1st and last question from window

        #mean reduce
        y_pd = tmp.sum(-1)[mask[:,1:-1]==1]/lens

        #switch prediction to question-based
        y_gt = kwargs[key_ktbench_label_seq][:, 1:-1]  #remove 1st question

        y_gt = y_gt[mask[:,1:-1]==1]

        return {
            'predict': y_pd,
            'target': y_gt,
            'len': len(y_pd)
        }


    def start(self):
        self.logs.train_starts(self.cfg.model_cls.__name__, self.traincfg, self.cfg)
        models = []
        test_logs = []
        for kfold in range(1, self.kfolds + 1):
            self.init_model()
            self.init_dataloader(kfold)
            print(f"[INFO] training start at kfold {kfold} out of {self.kfolds} folds...")
            print(f"-------")
            eval_logs = []
            AUCs = []
            best_auc = -1
            best_epoch = -1
            for epoch in range(1, self.n_epoch+1):
                losses = self.train(epoch)
                evals = self.evaluate(epoch)

                #log info
                #if not eval_logs:
                #    eval_logs['epoch'] = []
                #    eval_logs.update({k: [] for k in evals.keys()})
                evals['epoch'] = epoch
                eval_logs.append(evals)

                print(losses)
                print(evals)

                auc = evals['auc']
                AUCs.append(auc)
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    self.logs.save_best_model(self.model, best_epoch, kfold)
                    AUCs = []

                if len(AUCs) >= self.n_stop_check:
                    if max(AUCs) < best_auc:
                        print(f"[INFO] stopped training at epoch number {epoch}, no improvement in last {self.n_stop_check} epochs")
                        break
                    
            self.model = self.logs.load_best_model(self.model.__class__, kfold)
            tests = self.test(kfold)
            tests.update({'kfold': kfold, 
                          'best_epcoh': best_epoch,
                          'num_epochs': len(eval_logs)})
            test_logs.append(tests)
            print(tests)
            yamld.write_dataframe(self.logs.current_checkpoint_folder/f"valid_fold_{kfold}.yaml", pd.DataFrame(eval_logs))
            yamld.write_dataframe(self.logs.current_checkpoint_folder/f"test.yaml", pd.DataFrame(test_logs))


    def train(self, epoch_num):
        self.model.train()
        losses = []
        for batch_id, batch in enumerate(tqdm(self.train_dataloader, desc= f"[EPOCH={epoch_num}]")):
            pre_sum_losses = self.model.losses(**batch)
            loss = sum(pre_sum_losses.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        #validate epoch
        return {"loss_mean": sum(losses)/len(losses)}

    def evaluate(self, epoch_num):
        return self._evaluate(epoch_num, data_loader=self.valid_dataloader)

    def _evaluate(self, epoch_num, data_loader, description="[Inference]"):
        self.model.eval()
        preds = []
        trgts = []
        for batch_id, batch in enumerate(tqdm(data_loader, desc=description)):
            y_pd, idxslice = self.model.ktbench_predict(**batch)
            batch_eval = self.eval_method(y_pd, idxslice, self.cfg.dataset2model_feature_map, **batch)

            preds.append(batch_eval['predict'])
            trgts.append(batch_eval['target'])

        preds = torch.hstack(preds).cpu().detach().numpy()
        trgts = torch.hstack(trgts).cpu().detach().numpy()

        return compute_metrics(trgts, preds)
    
    def test(self, kfold):
        if not self.cfg.all_in_one:
            return self._evaluate(kfold, data_loader=self.test_dataloader, description=f"[Test fold {kfold}]")
        else:
            return self._all_in_one_test(kfold, data_loader=self.test_dataloader)


    def _all_in_one_eval(self, y_pd, idxslice, dataset2model_feature_map, **kwargs):
        key_exer_seq_mask = dataset2model_feature_map.get(*2*('ktbench_exer_seq_mask',)) 
        key_kc_seq_mask = dataset2model_feature_map.get(*2*('ktbench_kc_seq_mask',))
        key_unfold_seq_mask = dataset2model_feature_map.get(*2*('ktbench_unfold_seq_mask',))
        key_ktbench_label_seq = dataset2model_feature_map.get(*2*('ktbench_label_seq',))

        mask = kwargs[key_exer_seq_mask]
        kc_seq_mask = kwargs[key_kc_seq_mask]
        unfold_seq_mask = kwargs[key_unfold_seq_mask]
        tgt_index = kwargs['ktbench_allinone_tgt_index']
        label = kwargs['ktbench_allinone_label']
        ids = kwargs['ktbench_allinone_id']

        start =  idxslice.start if idxslice.start  else 0
        tgt_index += start

        #remove indices that was not calculated by the model
        max_len = y_pd.shape[-1] 
        tmpmask = tgt_index < max_len
        y_pd = y_pd[tmpmask, :] 
        tgt_index = tgt_index[tmpmask] 
        label = label[tmpmask]
        y_pd = y_pd.gather(dim=-1, index=tgt_index.unsqueeze(-1))
        ret  = {
            'predict': y_pd,
            'target': label,
            'ids': ids
        
        }
        return ret

    def _all_in_one_test(self, kfold, data_loader):
        self.model.eval()
        preds = {}
        trgts = {}
        for id, batch in enumerate(tqdm(data_loader, desc='all_in_one test fold {}'.format(kfold))):
            y_pd, idxslice = self.model.ktbench_predict(**batch)
            batch_eval = self._all_in_one_eval(y_pd, idxslice, self.cfg.dataset2model_feature_map, **batch)
            batch_ids = batch_eval['ids']
            batch_preds = batch_eval['predict']
            batch_tgt = batch_eval['target']
            preds.update({id: preds.get(id, []) + [pred] for id, pred in zip(batch_ids, batch_preds)})
            trgts.update(dict(zip(batch_ids, batch_tgt)))

        preds = map(lambda x: sum(x)/len(x), preds.values())
        trgts = list(trgts.values())
        preds = torch.hstack(list(preds)).cpu().detach().numpy()
        trgts = torch.hstack(trgts).cpu().detach().numpy()

        return compute_metrics(trgts, preds)
