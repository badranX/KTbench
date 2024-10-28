from noleak.run import bench_model
from dataclasses import dataclass

from noleak.model.dkt.dkt_ad import DKT_AD
from noleak.datapipeline.pipeline import Pipeline
from dataclasses import dataclass


def main(datasets=['assist2009', 'corr2_assist2009', 'duolingo2018_es_en', 'algebra2005', 'riiid2020']):
    @dataclass
    class Cfg:
        #dataset_name = "AKT_assist2017"
        #dataset_name = "assist2009"
        #dataset_name = "dualingo2018"
        dataset_name = "corr_assist2009"
        window_size: int = 150
        add_hide_label = False
        add_teacher_mask = True
        is_unfold = True
        is_unfold_fixed_window = False
        #eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        eval_method = Pipeline.EVAL_UNFOLD_REDUCE
        model_cls = DKT_AD
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 100
        lr = 0.001
    

    bench_model(Cfg(), Traincfg(), datasets = datasets)
