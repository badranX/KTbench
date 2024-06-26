from ktbench.run import bench_model
from dataclasses import dataclass
from ktbench.datapipeline.pipeline import Pipeline
from ktbench.model.dkt.mask_label_dkt import MaskedDKT
from ktbench.model.dkt.mask_label_dkt import Params


def main(datasets=['assist2009', 'corr_assist2009', 'duolingo2018_es_en']):
    @dataclass
    class Cfg:
        model_cls = MaskedDKT
        window_size: int = 150
        add_mask_label = True
        is_unfold = True

        eval_method = Pipeline.EVAL_UNFOLD_REDUCE
        kfold = 5

    @dataclass
    class Traincfg:
        batch_size = 128
        eval_batch_size = 128
        n_epoch = 100
        lr = 0.001

    
    prm = Params() 
    cfg = Cfg()
    prm.separate_qa = False
    cfg.append2logdir = '_seperate_qa_False'
    bench_model(cfg, Traincfg(), hyper_params=prm, datasets = datasets)

    print('start trining same model with _seperate_qa_True')
    prm = Params() 
    cfg = Cfg()
    prm.separate_qa = True
    cfg.append2logdir = '_seperate_qa_True'
    bench_model(cfg, Traincfg(), hyper_params=prm, datasets = datasets)
if __name__ == '__main__':
    main()