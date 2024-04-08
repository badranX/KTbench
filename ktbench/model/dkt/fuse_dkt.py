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


class DKT_Fuse(BaseModel):
    MODEL_FEATURE_MAP = {
        'ktbench_exer_seq': 'exer_seq' ,
        'ktbench_label_seq' : 'label_seq',
        'ktbench_kc_seq' : 'cpt_seq',
        'ktbench_kc_seq_mask' : 'cpt_seq_mask',
        'ktbench_exer_seq_mask': 'mask_seq'
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
        self.fc_layer = nn.Linear(self.prm.hidden_size, self.n_item+1)

    def forward(self, exer_seq, label_seq, cpt_seq, cpt_seq_mask, mask_seq, **kwargs):
        
        # obtain emb_q, emb_c, emb_qca, emb_qc
        k = self.exer_emb(cpt_seq + cpt_seq_mask*label_seq.unsqueeze(-1).long())
        k = k*cpt_seq_mask.unsqueeze(-1)
        emb_c = k.sum(-2)/(cpt_seq_mask.sum(-1) + (1-mask_seq)).unsqueeze(-1)
        emb_c = emb_c * mask_seq.unsqueeze(-1)
        #done summing embeddings

        input_x = emb_c
        exer_seq =  None
        output, _ = self.seq_model(input_x)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        
        #y_pd[...,0] = y_pd[...,0]*0.
        #y_pd[...,0] = 0.
        idxs = cpt_seq[:, 1:] + 1
        mask_res = cpt_seq_mask[:, 1:]
        idxs = mask_res*idxs
        y_pd = y_pd[:, :-1][torch.arange(idxs.shape[0])[...,None, None], torch.arange(idxs.shape[1])[None,...,None], idxs]
        lens = mask_res.sum(-1)
        y_pd = y_pd*mask_res
        y_pd = y_pd.sum(-1)

        ones= torch.ones_like(lens)
        lens = lens*mask_seq[:,1:] + ones*(1-mask_seq[:,1:])
        #y_pd = y_pd*mask_seq[:,1:]
        y_pd = y_pd/lens 
        return y_pd

    @torch.no_grad()
    def ktbench_predict(self, **kwargs):
        y_pd = self(**kwargs)
        #y_pd = y_pd[:, :-1]

        return y_pd, slice(1, None)

    def losses(self, **kwargs):
        y_pd = self(**kwargs)
        #y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt.float()
        )
        return {
            'loss_main': loss
        }