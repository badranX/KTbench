""" 
Note: we implement preprocessing here in order to minmise the change to the original AKT implementation
"""

from torch.utils.data import DataLoader
from ktbench.trainlogs import LogsHandler
import pandas as pd

from ktbench.model.akt.original_akt import AKT
from ktbench.train import Trainer
from ktbench.datapipeline.pipeline import Pipeline
from dataclasses import dataclass


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    ds = pipline.start(do_split=False)
    return pipline, ds
    
def fit(cfg, traincfg):
    trainer = Trainer(traincfg, cfg)
    trainer.start()

if __name__ == '__main__':
    IS_REDUCE_EVAL = True
    @dataclass
    class Cfg:
        #dataset_name = "AKT_assist2017"
        dataset_name = "assist2009"
        #dataset_name = "dualingo2018"
        #dataset_name = "corr_assist2009"
        multi2one_kcs = False
        window_size: int = 100
        is_unfold = True
        all_in_one = True
        #a must for simplekt
        #is_unfold_fixed_window = True
        eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        #eval_method = Trainer.EVAL_UNFOLD_REDUCE
        model_cls = AKT

        extra_features = {'qa_data': 'unfold',
                          'q_data': 'unfold',
                          'pid_data': 'unfold' }


    @dataclass
    class Traincfg:
        batch_size = 32
        eval_batch_size = 32
        n_epoch = 1
        lr = 0.001
    
    cfg = Cfg()

    pipline, ds = init_datapipeline(cfg) 

    def m(entry, ktbench_label_unfold_seq='ktbench_label_unfold_seq'):
        
        labels = entry[ktbench_label_unfold_seq]

        mask = entry['ktbench_unfold_seq_mask']
        pid_data = entry['ktbench_exer_unfold_seq'].clone()

        cpt = entry['ktbench_kc_unfold_seq'].clone()
        #AKT original model dont use 0s
        cpt = cpt + 1
        pid_data= pid_data + 1
        cpt[mask == 0] = 0
        pid_data[mask == 0] = 0
        #entry['ktbench_kc_unfold_seq'] = cpt
        #entry['ktbench_exer_unfold_seq'] = exer 
        entry['qa_data'] = (cpt + labels* cfg.n_kc).int()
        entry['q_data'] = cpt
        entry['pid_data'] = pid_data

        #labels[mask == 0]= -1
        #entry['ktbench_label_unfold_seq'] = labels
        return entry

    ds = ds.map(m, batched=False)
    pipline.split_dataset(ds)
    cfg.test_ds = cfg.test_ds.map(lambda x: m(x, ktbench_label_unfold_seq='target'), batched=False)

    fit(cfg, Traincfg())