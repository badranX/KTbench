from noleak.run import bench_model
from dataclasses import dataclass
from noleak.datapipeline.pipeline import Pipeline
from noleak.model.akt.mask_label_akt import AKT_ML


def main(datasets=['assist2009', 'corr_assist2009', 'duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = AKT_ML
        window_size: int = 150
        add_mask_label = True
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