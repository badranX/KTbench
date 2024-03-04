import torch.nn as nn
import torch

from .utils.common import xavier_normal_initialization, xavier_uniform_initialization, kaiming_normal_initialization, kaiming_uniform_initialization
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
        

    def apply(self, fn):
        #adapted from EduStudio
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self


    def _init_params(self):
        #Adapted from Edustudio
        """Initialize the model parameters
        """
        if not hasattr(self.cfg, 'param_init_type'):
            self.cfg.param_init_type = 'xavier_normal'

        if self.cfg.param_init_type == 'default':
            pass
        elif self.cfg.param_init_type  == 'xavier_normal':
            self.apply(xavier_normal_initialization)
        elif self.cfg.param_init_type  == 'xavier_uniform':
            self.apply(xavier_uniform_initialization)
        elif self.cfg.param_init_type  == 'kaiming_normal':
            self.apply(kaiming_normal_initialization)
        elif self.cfg.param_init_type  == 'kaiming_uniform':
            self.apply(kaiming_uniform_initialization)
        elif self.cfg.param_init_type  == 'init_from_pretrained':
            self._load_params_from_pretrained()