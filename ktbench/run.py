from ktbench.train import Trainer
from ktbench.datapipeline.pipeline import Pipeline


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    pipline.start()
    return pipline
    
def run_trainer(cfg, traincfg, hyper_params=None, start_from_kfold=1):
    trainer = Trainer(traincfg, cfg, hyper_params)
    trainer.start(start_from_kfold)
    return trainer

def bench_model(cfg, traincfg, datasets=None, hyper_params=None):
    if hasattr(cfg, 'dataset_name') and datasets:
        print("[WARNING] config contain dataset_name but datasets were provided. Will default to the provided datasets")
    if not datasets:
        datasets = [cfg.dataset_name]
        
    trainers = []
    for ds in datasets:
        if isinstance(ds, tuple):
            kfold_start = ds[0]
            ds = ds[1]
        else:
            kfold_start = 1
        print("training model:", cfg.model_cls.__name__)
        print("start training dataset", ds)
        cfg.dataset_name = ds 
        pipeline = init_datapipeline(cfg)
        trainers.append( run_trainer(cfg, traincfg, hyper_params, kfold_start))
                       
    return trainers, pipeline
