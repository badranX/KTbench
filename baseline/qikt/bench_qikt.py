from ktbench.run import bench_model
from dataclasses import dataclass

from ktbench.model.qikt.qikt import QIKT

from ktbench.datapipeline.pipeline import Pipeline
def main(datasets=['assist2009', 'algebra2005', 'duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = QIKT
        window_size: int = 150
        is_unfold = False

        eval_method = Pipeline.EVAL_QUESTION_LEVEL
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 24
        eval_batch_size = 24
        n_epoch = 100
        lr = 0.001
    
    bench_model(Cfg(), Traincfg(), datasets = datasets)

if __name__ == '__main__':
    main()