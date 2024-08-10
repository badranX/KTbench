import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass

if __name__ == "__main__":
    import importlib.machinery
    basemodel = importlib.machinery.SourceFileLoader('basemodel','../basemodel.py').load_module()
    BaseModel = basemodel.BaseModel
else:
    from ..basemodel import BaseModel


@dataclass
class Params:
    emb_size = 128
    hidden_size = 128
    num_layers = 1
    dropout_rate = 0.2
    rnn_or_lstm = 'lstm'


class DKT(BaseModel):
    MODEL_FEATURE_MAP = {
         'ktbench_kc_unfold_seq' : 'exer_seq' ,
         'ktbench_unfold_seq_mask' : 'mask_seq' ,
         'ktbench_label_unfold_seq' : 'label_seq' ,
        }
    def __init__(self, cfg, params=Params()):
        super().__init__(cfg, params)

        self.n_item = self.n_kc
        self.build_model()

    def build_cfg(self):
        assert self.prm.rnn_or_lstm in {'rnn', 'lstm'}

    def build_model(self):
        self.exer_emb = nn.Embedding(
            self.n_item * 2, self.prm.emb_size
        )
        if self.prm.rnn_or_lstm == 'rnn':
            self.seq_model = nn.RNN(
                self.prm.emb_size, self.prm.hidden_size, 
                self.prm.num_layers, batch_first=True
            )
        else:
            self.seq_model = nn.LSTM(
                self.prm.emb_size, self.prm.hidden_size, 
                self.prm.num_layers, batch_first=True
            )
        self.dropout_layer = nn.Dropout(self.prm.dropout_rate)
        self.fc_layer = nn.Linear(self.prm.hidden_size, self.n_item)

    def forward(self, exer_seq, label_seq, **kwargs):
        maxer = exer_seq.max().item()
        input_x = self.exer_emb(exer_seq + label_seq.long() * self.n_item)
        output, _ = self.seq_model(input_x)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        return y_pd

    @torch.no_grad()
    def ktbench_predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)

        return y_pd, slice(1, None)

    def losses(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }