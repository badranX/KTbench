from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.datapipeline.pipeline import Pipeline
from ktbench.model.akt.extra_mask_akt import ExtraMaskAKT


def main(datasets=['corr_assist2009', 'duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = ExtraMaskAKT
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