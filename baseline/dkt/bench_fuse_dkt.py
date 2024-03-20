from ktbench.run import bench_model
from dataclasses import dataclass

from ktbench.model.dkt.fuse_dkt import FuseDKT

from ktbench.datapipeline.pipeline import Pipeline
def main(datasets=['assist2009', 'corr2_assist2009', 'duolingo2018_es_en', 'algebra2005', 'riiid2020']):
    @dataclass
    class Cfg:
        model_cls = FuseDKT
        window_size: int = 150
        is_unfold = False

        eval_method = Pipeline.EVAL_QUESTION_LEVEL
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 100
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = datasets)

if __name__ == '__main__':
    main()