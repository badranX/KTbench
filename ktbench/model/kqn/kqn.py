# -----------------------------------------------------------------------------
# This file was adapted from EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
    rnn_hidden_size = 128
    mlp_hidden_size = 128
    n_rnn_layers = 1
    rnn_or_lstm = 'lstm'
    dropout = 0.2

class KQN(BaseModel):
    MODEL_FEATURE_MAP = {
         'ktbench_kc_unfold_seq' : 'exer_seq' ,
         'ktbench_unfold_seq_mask' : 'mask_seq' ,
         'ktbench_label_unfold_seq' : 'label_seq' ,
        }
    def __init__(self, cfg, params=Params()):
        super().__init__(cfg, params)

        self.n_item = self.n_kc
        self.build_model()
    
    
    def build_model(self):
        # helper variable for making a one-hot vector for rnn input
        self.inter_emb = torch.eye(self.n_item * 2).to(self.device)
        # helper variable for making a one-hot vector for skills
        self.cpt_emb = torch.eye(self.n_item).to(self.device)

        if self.prm.rnn_or_lstm == 'gru':
            self.seq_model = nn.GRU(
                2 * self.n_item, self.prm.rnn_hidden_size, 
                self.prm.n_rnn_layers, batch_first=True
            )
        else:
            self.seq_model = nn.LSTM(
                2 * self.n_item, self.prm.rnn_hidden_size, 
                self.prm.n_rnn_layers, batch_first=True
            )
        
        self.skill_encoder = nn.Sequential(
            nn.Linear(self.n_item, self.prm.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.prm.mlp_hidden_size, self.prm.emb_size),
            nn.ReLU()
        )
        self.fc_layer = nn.Linear(self.prm.rnn_hidden_size, self.prm.emb_size)
        self.drop_layer = nn.Dropout(self.prm.dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, exer_seq, label_seq, **kwargs):
        in_data = self.inter_emb[(exer_seq[:,:-1] + label_seq[:,:-1] * self.n_item).long()]
        next_skills = self.cpt_emb[(exer_seq[:,1:]).long()]

        encoded_knowledge = self.encode_knowledge(in_data.to(self.device))
        encoded_skills = self.encode_skills(next_skills.to(self.device))
        encoded_knowledge = self.drop_layer(encoded_knowledge)

        y_pd = torch.sum(encoded_knowledge * encoded_skills, dim=2) # (batch_size, max_seq_len)
        y_pd = self.sigmoid(y_pd)

        return y_pd

    def encode_knowledge(self, in_data):
        batch_size = in_data.size(0)
        self.hidden = self.init_hidden(batch_size)
        
        # rnn_input = pack_padded_sequence(in_data, seq_len, batch_first=True)
        rnn_output, _ = self.seq_model(in_data, self.hidden)
        # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True) # (batch_size, max_seq_len, n_rnn_hidden)
        encoded_knowledge = self.fc_layer(rnn_output) # (batch_size, max_seq_len, n_hidden)

        return encoded_knowledge
    
    def encode_skills(self, next_skills):
        encoded_skills = self.skill_encoder(next_skills) # (batch_size, max_seq_len, n_hidden)
        encoded_skills = F.normalize(encoded_skills, p=2, dim=2) # L2-normalize

        return encoded_skills

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        if self.prm.rnn_or_lstm == 'lstm':
            return (Variable(weight.new(self.prm.n_rnn_layers, batch_size, self.prm.rnn_hidden_size).zero_()),
                    Variable(weight.new(self.prm.n_rnn_layers, batch_size, self.prm.rnn_hidden_size).zero_()))
        else:
            return Variable(weight.new(self.prm.n_rnn_layers, batch_size, self.prm.rnn_hidden_size).zero_())

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd[:-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=1
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def ktbench_predict(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd[:-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=1
        # ).squeeze(dim=-1)
        return y_pd, slice(1, None)

    def losses(self, **kwargs):
        y_pd = self(**kwargs)
        # y_pd = y_pd.gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }


