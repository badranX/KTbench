from noleak.train import Trainer
from noleak.datapipeline.pipeline import Pipeline


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    pipline.start()
    return pipline
    
def run_trainer(cfg, traincfg, hyper_params=None):
    trainer = Trainer(traincfg, cfg, hyper_params)
    trainer.start()
    return trainer

def bench_model(cfg, traincfg, datasets=None, hyper_params=None):
    if hasattr(cfg, 'dataset_name') and datasets:
        print("[WARNING] config contain dataset_name but datasets were provided. Will default to the provided datasets")
    if not datasets:
        datasets = [cfg.dataset_name]
        
    for ds in datasets:
        print("training model:", cfg.model_cls.__name__)
        print("start training dataset", ds)
        cfg.dataset_name = ds 
        init_datapipeline(cfg)
        run_trainer(cfg, traincfg, hyper_params)
