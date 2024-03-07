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
        add_mask_label = True
        add_teacher_mask = True
        is_unfold = True
        is_unfold_fixed_window = False
        all_in_one = False
        #eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        eval_method = Trainer.EVAL_UNFOLD_REDUCE

        splits = [0.8, 0.3] 


    @dataclass
    class Traincfg:
        batch_size = 32
        eval_batch_size = 32
        n_epoch = 1
        lr = 0.001
    
    
    bench_model(Cfg(), Traincfg(), datasets = ['assist2009', 'corr_assist2009'])