from ktbench.run import bench_model
from dataclasses import dataclass

from ktbench.model.dkt.dkt import DKT

from ktbench.datapipeline.pipeline import Pipeline
def main(datasets=['assist2009']):
    @dataclass
    class Cfg:
        model_cls = DKT
        window_size: int = 150
        is_unfold = True
        all_in_one = True
        is_test_all_in_one = True
        multi2one_kcs = True

        eval_method = Pipeline.EVAL_UNFOLD_REDUCE
        kfold = 1

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 2
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = datasets)

if __name__ == '__main__':
    main()