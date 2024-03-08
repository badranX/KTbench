from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.train import Trainer
from ktbench.model.dkt.dkt import DKT


if __name__ == '__main__':
    IS_REDUCE_EVAL = True
    @dataclass
    class Cfg:
        model_cls = DKT
        #dataset_name = "AKT_assist2017"
        #dataset_name = "assist2009"
        #dataset_name = "dualingo2018_es_en"
        #dataset_name = "corr_assist2009"
        multi2one_kcs = True
        window_size: int = 100
        add_mask_label = False
        add_teacher_mask = False
        is_unfold = True
        is_unfold_fixed_window = False
        all_in_one = True
        #eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        eval_method = Trainer.EVAL_UNFOLD_REDUCE

        splits = [0.6, 0.5] 


    @dataclass
    class Traincfg:
        batch_size = 256
        eval_batch_size = 128
        n_epoch = 10
        lr = 0.001
    
    #bench_model(Cfg(), Traincfg(), datasets = ['assist2009', 'corr_assist2009', 'dualingo2008_es_en'])
    bench_model(Cfg(), Traincfg(), datasets = ['corr_assist2009'])
    
    

    
