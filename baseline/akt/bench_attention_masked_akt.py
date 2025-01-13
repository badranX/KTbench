from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.train import Trainer
from ktbench.model.akt.attention_masked_akt import AKT_QM


def main(datasets=['duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = AKT_QM
        window_size: int = 150
        is_attention = True
        is_unfold = True

        #eval_method = Trainer.EVAL_UNFOLD_REDUCE
        eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 32
        eval_batch_size = 32
        n_epoch = 100
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = ['assist2009', 'corr_assist2009', 'duolingo2018_es_en'])
    

if __name__ == '__main__':
    main()