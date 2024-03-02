from torch.utils.data import DataLoader
from ktbench.trainlogs import LogsHandler
import pandas as pd

from ktbench.model.dkt.selfteach_dkt import SelfTeachDKT
from ktbench.train import Trainer
from ktbench.datapipeline.pipeline import Pipeline
from dataclasses import dataclass


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    pipline.start(gen=None, from_middata=False)
    return pipline
    
def fit(cfg, traincfg):
    trainer = Trainer(traincfg, cfg)
    trainer.start()

if __name__ == '__main__':
    IS_REDUCE_EVAL = True
    @dataclass
    class Cfg:
        #dataset_name = "AKT_assist2017"
        #dataset_name = "assist2009"
        #dataset_name = "dualingo2018"
        dataset_name = "corr_assist2009"
        window_size: int = 100
        add_hide_label = False
        add_teacher_mask = True
        is_unfold = True
        is_unfold_fixed_window = False
        #eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        eval_method = Trainer.EVAL_UNFOLD_REDUCE
        model_cls = SelfTeachDKT


    @dataclass
    class Traincfg:
        batch_size = 32
        eval_batch_size = 32
        n_epoch = 40
        lr = 0.001
    
    cfg = Cfg()

    pipline = init_datapipeline(cfg) 
    eval_methods = [Trainer.EVAL_UNFOLD_KC_LEVEL, 
                Trainer.EVAL_UNFOLD_KC_LEVEL]
    cfg.eval_method = eval_methods[0]

    fit(cfg, Traincfg())