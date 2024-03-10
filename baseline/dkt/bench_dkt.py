from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.train import Trainer
from ktbench.model.dkt.dkt import DKT


if __name__ == '__main__':
    @dataclass
    class Cfg:
        model_cls = DKT
        window_size: int = 150
        is_unfold = True
        all_in_one = True

        eval_method = Trainer.EVAL_UNFOLD_REDUCE
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 100
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = ['assist2009', 'corr_assist2009', 'duolingo2008_es_en'])