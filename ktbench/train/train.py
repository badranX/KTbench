from torch import autograd

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


class Collate:
    def __init__(self, seqs=None, seqofseq=None):
        self.seqs = seqs

    def pad_collate(self, batch):

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


class Trainer():

    EVAL_QUESTION_LEVEL, \
    EVAL_UNFOLD_KC_LEVEL, \
    EVAL_UNFOLD_REDUCE, \
    EVAL_UNFOLD_STEP, \
    *_ = range(10)
    
    TEST_LIKE_EVAL, \
    TEST_STEP, \
    *_ = range(10)

    def __init__(self, traincfg, cfg):
        self.inference_methods = {
            self.EVAL_QUESTION_LEVEL: self.question_eval,
            self.EVAL_UNFOLD_REDUCE: self.reduce_eval,
            self.EVAL_UNFOLD_KC_LEVEL: self.kc_eval,
        }
        model = cfg.model_cls(cfg).to(cfg.device)
        traincfg.model = model
        self.is_unfold = cfg.is_unfold
        self.logs = LogsHandler(cfg)
        cfg.logs = self.logs
        self.cfg = cfg
        self.device =cfg.device
        self.traincfg = traincfg
        self.is_padded = False if not hasattr(traincfg, 'is_padded') else traincfg.is_padded
        self.is_step_test = False if not hasattr(traincfg, 'is_step_test') else traincfg.is_step_test
        self.kfolds = 1 if not hasattr(cfg, 'kfold') else cfg.kfold
        self.model = self.traincfg.model
        self.n_stop_check = 10  if not hasattr(traincfg,'n_stop_check') else traincfg.n_stop_check
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

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.traincfg.lr, betas=(0.9, 0.999), eps=1e-8)
        
    def init_dataloader(self,k):
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
        clt_train = Collate(seqs = seqs_train)
        clt_test = Collate(seqs = seqs_test)
        clt_valid = Collate(seqs = seqs_valid)
        self.train_dataloader = DataLoader(self.cfg.train_ds[k], batch_size=self.traincfg.batch_size, collate_fn=clt_train.pad_collate)
        self.valid_dataloader = DataLoader(self.cfg.valid_ds[k], batch_size=self.traincfg.eval_batch_size, collate_fn=clt_valid.pad_collate)
        self.test_dataloader = DataLoader(self.cfg.test_ds, batch_size=self.traincfg.eval_batch_size, collate_fn= clt_test.pad_collate)


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

        assert (not idxslice.start or idxslice.start <=1) and (not idxslice.stop or idxslice.stop >= -1)
        start = 1 if idxslice.start is None else None
        stop = -1 if idxslice.stop is None else None
        y_pd = y_pd[...,start:stop]
        y_pd = y_pd[mask[...,1:-1] == 1]

        y_gt = kwargs[key_ktbench_label_unfold_seq][:, 1:-1]  #remove 1st question

        y_gt = y_gt[mask[:,1:-1]==1]
        
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

        #reset to normal length, assuming we remove first & last question
        if True or idxslice != slice(None,None):
            #TODO optimize!
            new_ypd = -1*torch.ones_like(unfold_seq_mask, dtype=y_pd.dtype)
            new_ypd[...,idxslice] = y_pd
            y_pd = new_ypd

        tmp = torch.zeros(*kc_seq_mask.shape, dtype=y_pd.dtype).to(self.device)
        #todo make sure masked exersies, is treated as exer 0 and mapped in kc_seq_mask
        tmp[kc_seq_mask==1] = y_pd[unfold_seq_mask == 1]
        #TODO adjust this
        tmp = tmp[:,1:-1]  #remove 1st question

        lens = kc_seq_mask[:,1:-1].sum(-1)[mask[:,1:-1]==1]

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
        AUCs = []
        models = []
        best_auc = -1
        best_epoch = -1

        eval_logs = {}
        for kfold in range(1, self.kfolds + 1):
            self.init_dataloader(kfold)
            print(f"[INFO] training start at kfold {kfold} out of {self.kfolds} folds...")
            print(f"-------")
            for epoch in range(1, self.n_epoch+1):
                losses = self.train(epoch)
                evals = self.evaluate(epoch)

                #log info
                #if not eval_logs:
                #    eval_logs['epoch'] = []
                #    eval_logs.update({k: [] for k in evals.keys()})
                evals['epoch'] = epoch
                eval_logs.update({k: eval_logs.get(k,[]) + [v] for k, v in evals.items()})

                print(losses)
                print(evals)

                auc = evals['auc']
                AUCs.append(auc)
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    self.logs.save_best_model(self.model, best_epoch)
                    AUCs = []

                if len(AUCs) >= self.n_stop_check:
                    if max(AUCs) < best_auc:
                        print(f"[INFO] stopped training at epoch number {epoch}, no improvement in last {self.n_stop_check} epochs")
                        break
            if not self.is_step_test:
                #tests = self.test(kfold)
                #print("Test fold {}:".format(kfold), tests)
                pass
            else:
                pass

        yamld.write_dataframe(self.logs.current_checkpoint_folder/"evals.yaml", pd.DataFrame(eval_logs))


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
        
        for batch_id, batch in enumerate(tqdm(data_loader, desc=description)):
            preds = []
            trgts = []
            y_pd, idxslice = self.model.ktbench_predict(**batch)
            batch_eval = self.eval_method(y_pd, idxslice, self.cfg.dataset2model_feature_map, **batch)

            preds.append(batch_eval['predict'])
            trgts.append(batch_eval['target'])

        preds = torch.hstack(preds).cpu().detach().numpy()
        trgts = torch.hstack(trgts).cpu().detach().numpy()

        return {"auc": compute_auc(trgts, preds)}
    
    def test(self, kfold):
        return self._evaluate(kfold, data_loader=self.test_dataloader, description="[Test]")


    def step_unfold_test(self, fold_num):
        extras = {} if not hasattr(self.cfg, 'extra_sequence_axis') else self.cfg.extra_sequence_axis
        #sequence_axis = dict(SEQUENCE_AXIS)
        sequence_axis = extras
        sequence_axis.update(extras)
        d2m = self.cfg.dataset2model_feature_map
        sequence_axis = {model_feature: sequence_axis[ktbench_feature] for ktbench_feature, model_feature in d2m.items()}

        unfold_seq_mask = d2m.get(*2*('ktbench_unfold_seq_mask',)) 
        key_ktbench_label_unfold_seq = d2m.get(*2*('ktbench_label_unfold_seq',))

        for batch_id, batch in enumerate(tqdm(self.test_dataloader, desc="[TEST]")):
            mask = batch[unfold_seq_mask]
            for i in range(2, mask.shape[-1]+1):
                cutbatch = dict(batch)
                tmp = {k: v[:i] for k,v in sequence_axis if k in batch}
                cutbatch.update(tmp)
                y_pd, idxslice = self.model.ktbench_predict(**cutbatch)
                #TODO
                batch_eval = self.eval_method(y_pd, idxslice, self.cfg.dataset2model_feature_map, **batch)

            preds.append(batch_eval['predict'])
            trgts.append(batch_eval['target'])

        preds = torch.hstack(preds).cpu().detach().numpy()
        trgts = torch.hstack(trgts).cpu().detach().numpy()

        return {"auc": compute_auc(trgts, preds)}
