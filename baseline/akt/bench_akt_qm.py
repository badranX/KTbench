from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.datapipeline.pipeline import Pipeline
from ktbench.model.akt.question_masked_akt import AKT_QM


def main(datasets=['duolingo2018_es_en', 'corr2_assist2009', 'assist2009', 'algebra2005', 'riiid2020']):
    @dataclass
    class Cfg:
        model_cls = AKT_QM
        window_size: int = 150
        is_attention = True
        is_unfold = True

        eval_method = Pipeline.EVAL_UNFOLD_REDUCE
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