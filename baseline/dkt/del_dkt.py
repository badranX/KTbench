from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.datapipeline.pipeline import Pipeline
from ktbench.model.dkt.dkt import DKT

def main(datasets=['corr_assist2009', 'duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = DKT
        window_size: int = 150
        is_unfold = True
        all_in_one = True

        #eval_method = Pipeline.EVAL_UNFOLD_REDUCE
        eval_method = Pipeline.EVAL_UNFOLD_KC_LEVEL
        kfold = 1

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 100
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = datasets)

if __name__ == '__main__':
    main()
