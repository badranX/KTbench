from ktbench.train import Trainer
from ktbench.datapipeline.pipeline import Pipeline


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    pipline.start()
    return pipline
    
def run_trainer(cfg, traincfg):
    trainer = Trainer(traincfg, cfg)
    trainer.start()
    return trainer

def bench_model(cfg, traincfg, datasets=None):
    if not datasets:
        datasets = [cfg.dataset_name]
        
    for ds in datasets:
        print("start training dataset", ds)
        cfg.dataset_name = ds 
        init_datapipeline(cfg)
        run_trainer(cfg, traincfg)
