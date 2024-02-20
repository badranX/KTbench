import torch.nn as nn
import torch

from dataclasses import dataclass

@dataclass
class Params:
    l2: float = 1e-5
    kq_same: float = 1,
    dropout_rate: float = 0.05,
    separate_qa: bool = False,
    d_model: float = 256,
    n_blocks: float =1,
    final_fc_dim: float = 512,
    n_heads: float = 8,
    d_ff: float = 2048,

class BaseModel(nn.Module):
    def __init__(self, cfg, params):
        super().__init__()
        self.prm = params
        self.cfg = cfg
        self.device = self.cfg.device
        self.set_dataset_info()
        self.is_default_eval = True
        
        
    def set_dataset_info(self):
        self.n_stu = self.cfg.n_stu
        self.n_exer = self.cfg.n_exer
        self.n_kc = self.cfg.n_kc
