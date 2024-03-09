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
    separate_qa = True



class MaskedDKT(BaseModel):

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
            self.n_item , self.prm.emb_size
        )

        if self.prm.separate_qa:
            self.answer_emb = nn.Embedding(
                self.n_item * 3, self.prm.emb_size
            )
        else:
            self.answer_emb = nn.Embedding(
                3, self.prm.emb_size
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
        label_seq = kwargs['ktbench_masked_label_unfold_seq']
        maxer = exer_seq.max().item()
        #input_x = self.exer_emb(exer_seq + label_seq.long() * self.n_item)
        if self.prm.separate_qa:
            input_x = self.answer_emb(exer_seq + label_seq.long()* self.n_item)
        else:
            answers = self.answer_emb(label_seq.long())
            input_x = self.exer_emb(exer_seq) + answers 
        output, _ = self.seq_model(input_x)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        return y_pd
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)

        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'predict': y_pd,
            'target': y_gt
        }

    @torch.no_grad()
    def ktbench_trace(self, **kwargs):
        y_pd = self(**kwargs)
        return y_pd

    @torch.no_grad()
    def ktbench_predict(self, **kwargs):
        y_pd = self(**kwargs)

        #first and last question will be removed
        #tmp_pd = y_pd[...,0]
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        #mask
        #tmp_pd[:,0] = -1
        #y_pd = tmp_pd
        return y_pd, slice(1, None)

    @torch.no_grad()
    def reduce_predict_batch(self, **kwargs):
        key_exer_seq_mask = self.cfg.badkeys2features.get(*2*('ktbench_exer_seq_mask',)) 
        key_cpt_seq_mask = self.cfg.badkeys2features.get(*2*('ktbench_cpt_seq_mask',))
        key_unfold_seq_mask = self.cfg.badkeys2features.get(*2*('ktbench_unfold_seq_mask',))
        key_ktbench_label_seq = self.cfg.badkeys2features.get(*2*('ktbench_label_seq',))

        y_pd = self(**kwargs)
        #first and last question will be removed
        tmp_pd = y_pd[...,0]
        tmp_pd[:, 1:] = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        y_pd = tmp_pd
        #y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]

        mask = kwargs[key_exer_seq_mask]
        cpt_seq_mask = kwargs[key_cpt_seq_mask]
        unfold_seq_mask = kwargs[key_unfold_seq_mask]

        tmp = torch.zeros(*cpt_seq_mask.shape, dtype=y_pd.dtype).to(self.device)
        #todo make sure masked exersies, is treated as exer 0 and mapped in cpt_seq_mask
        tmp[cpt_seq_mask==1] = y_pd[unfold_seq_mask == 1]
        tmp = tmp[:,1:-1]  #remove 1st question

        lens = cpt_seq_mask[:,1:-1].sum(-1)[mask[:,1:-1]==1]

        #mean reduce
        y_pd = tmp.sum(-1)[mask[:,1:-1]==1]/lens

        #switch prediction to question-based
        y_gt = kwargs[key_ktbench_label_seq][:, 1:-1]  #remove 1st question

        y_gt = y_gt[mask[:,1:-1]==1]
        

        return {
            'predict': y_pd,
            'target': y_gt,
            'len': len(y_pd)
        }


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