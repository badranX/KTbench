from torch.utils.data import DataLoader

from ktbench.model.dkt.dkt import DKT
from ktbench.train import Trainer
from ktbench.datapipeline.pipeline import Pipeline
from dataclasses import dataclass

IS_REDUCE_EVAL = True

@dataclass
class Cfg:
    #yaml_middata='./test.yaml'
    #yaml_middata = '../AKT_assist2017_pid.yaml'
    window_size: int = 100
    is_pad_unfold = True
    #dataset_name = "assist2009"
    #dataset_name = "assist2017"
    #dataset_name = "assist2017"
    dataset_name = "AKT_assist2017"
    is_unfold = True
    is_unfold_fixed_window = True
    is_reduce_eval = IS_REDUCE_EVAL
    dataset2model_feature_map = {
     'ktbench_kc_unfold_seq' : 'exer_seq' ,
     'ktbench_unfold_seq_mask' : 'mask_seq' ,
     'ktbench_label_unfold_seq' : 'label_seq' ,
    }

cfg = Cfg()
pipline = Pipeline(cfg)
pipline.start(gen=None, from_middata=False)


model = DKT(cfg).to(cfg.device)


@dataclass
class Traincfg:
    #n_stop_check = 2
    eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
    batch_size = 32
    eval_batch_size = 32
    n_epoch = 2
    model = model
    lr = 0.001

trainer = Trainer(Traincfg(), cfg)
trainer.start()
    
m = trainer.logs.load_best_model(DKT)
print(m)