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
    emb_size = 100
    hidden_size = 100
    mlp_layer_num = 1
    dropout_rate = 0.2
    output_c_all_lambda = 1.0
    output_c_next_lambda = 1.0
    output_q_all_lambda = 1.0
    output_q_next_lambda = 0.0
    loss_c_all_lambda = 0.0
    loss_c_next_lambda = 0.0
    loss_q_all_lambda = 0.0
    loss_q_next_lambda = 0.0
    #output_mode = 'an_irt'
    output_mode = 'an_irt'


class QIKT(BaseModel):
    MODEL_FEATURE_MAP = {
        'ktbench_exer_seq': 'exer_seq' ,
        'ktbench_label_seq' : 'label_seq',
        'ktbench_kc_seq' : 'cpt_seq',
        'ktbench_kc_seq_mask' : 'cpt_seq_mask',
        'ktbench_exer_seq_mask': 'mask_seq'
    }

    def __init__(self, cfg, params=Params()):
        super().__init__(cfg, params)
        self.n_item = self.n_exer
        self.n_cpt = self.n_kc
        self.build_cfg()
        self.build_model()

    def build_cfg(self):
        self.output_c_all_lambda = self.prm.output_c_all_lambda
        self.output_c_next_lambda = self.prm.output_c_next_lambda
        self.output_q_all_lambda = self.prm.output_q_all_lambda
        self.output_q_next_lambda = self.prm.output_q_next_lambda
        self.loss_c_all_lambda = self.prm.loss_c_all_lambda
        self.loss_c_next_lambda = self.prm.loss_c_next_lambda
        self.loss_q_all_lambda = self.prm.loss_q_all_lambda
        self.loss_q_next_lambda = self.prm.loss_q_next_lambda
        self.output_mode = self.prm.output_mode

    def build_model(self):
        num_q, num_c = self.n_item, self.n_cpt
        self.exer_emb = nn.Embedding(
            self.n_item, self.prm.emb_size
        )
        self.cpt_emb = nn.Embedding(
            self.n_cpt, self.prm.emb_size
        )
        self.que_lstm_layer = nn.LSTM(self.prm.emb_size * 4, self.prm.hidden_size, batch_first=True)
        self.concept_lstm_layer = nn.LSTM(self.prm.emb_size * 2, self.prm.hidden_size, batch_first=True)

        self.dropout_layer = nn.Dropout(self.prm.dropout_rate)

        self.out_question_next = MLP(self.prm.mlp_layer_num, self.prm.hidden_size * 3, 1, self.prm.dropout_rate)
        self.out_question_all = MLP(self.prm.mlp_layer_num, self.prm.hidden_size, num_q, self.prm.dropout_rate)

        self.out_concept_next = MLP(self.prm.mlp_layer_num, self.prm.hidden_size * 3, num_c, self.prm.dropout_rate)
        self.out_concept_all = MLP(self.prm.mlp_layer_num, self.prm.hidden_size, num_c, self.prm.dropout_rate)

        self.que_disc = MLP(self.prm.mlp_layer_num, self.prm.hidden_size * 2, 1, self.prm.dropout_rate)

    def forward(self, exer_seq, label_seq, cpt_seq, cpt_seq_mask, mask_seq, **kwargs):

        # obtain emb_q, emb_c, emb_qca, emb_qc
        emb_q = self.exer_emb(exer_seq)
        k = self.cpt_emb(cpt_seq)
        k = k*cpt_seq_mask.unsqueeze(-1)
        emb_c = k.sum(-2)/(cpt_seq_mask.sum(-1) + (1-mask_seq)).unsqueeze(-1)
        emb_c = emb_c * mask_seq.unsqueeze(-1)

       # emb_c = torch.sum(k * (cpt_seq_mask.unsqueeze(3).repeat(1, 1, 1, self.modeltpl_cfg['emb_size'])),
       #                   dim=2) / cpt_seq_mask.sum(dim=2).unsqueeze(2).repeat(1, 1, self.modeltpl_cfg['emb_size'])


        emb_qc = torch.cat((emb_q, emb_c), dim=2)
        mask_e = label_seq.unsqueeze(-1).repeat(1, 1, emb_qc.shape[-1]).to(torch.float)
        emb_qca = torch.cat((mask_e*emb_qc, (1-mask_e)*emb_qc), dim=-1)

        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]
        # question model
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = self.get_outputs(emb_qc_shift, que_h, data=exer_seq[:, 1:], add_name="", modeltpl_type="question")
        outputs = que_outputs

        # concept model
        emb_ca = torch.cat([emb_c.mul((1 - label_seq).unsqueeze(-1).repeat(1, 1, self.prm.emb_size)),
                            emb_c.mul((label_seq).unsqueeze(-1).repeat(1, 1, self.prm.emb_size))], dim=-1)  # s_t 扩展，分别对应正确的错误的情况

        emb_ca_current = emb_ca[:, :-1, :]
        # emb_c_shift = emb_c[:,1:,:]
        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = self.get_outputs(emb_qc_shift, concept_h, data=(cpt_seq[:, 1:, :], cpt_seq_mask[:, 1:, :]), add_name="", modeltpl_type="concept")
        outputs['y_concept_all'] = concept_outputs['y_concept_all']
        outputs['y_concept_next'] = concept_outputs['y_concept_next']


        if self.output_mode=="an_irt":
            def sigmoid_inverse(x,epsilon=1e-8):
                return torch.log(x/(1-x+epsilon)+epsilon)
            y = sigmoid_inverse(outputs['y_question_all'])*self.output_q_all_lambda + \
                sigmoid_inverse(outputs['y_concept_all'])*self.output_c_all_lambda + \
                sigmoid_inverse(outputs['y_concept_next'])*self.output_c_next_lambda
            y = torch.sigmoid(y)
        else:
            # output weight
            y = outputs['y_question_all'] * self.output_q_all_lambda \
                + outputs['y_concept_all'] * self.output_c_all_lambda \
                + outputs['y_concept_next'] * self.output_c_next_lambda
            y = y/(self.output_q_all_lambda + self.output_c_all_lambda + self.output_c_next_lambda)
        outputs['y'] = y
        return outputs

    @torch.no_grad()
    def ktbench_predict(self, **kwargs):
        outputs = self(**kwargs)
        y_pd = outputs['y']
        # y_pd = y_pd[:, :-1].gather(
        #     index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        # ).squeeze(dim=-1)
        #y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        return y_pd, slice(1, None)

    def losses(self, **kwargs):
        outputs = self(**kwargs)

        rshft = kwargs['label_seq'][:, 1:]
        maskshft = kwargs['mask_seq'][:, 1:].bool()
        # all
        loss_q_all = self.get_loss(outputs['y_question_all'],rshft,maskshft)
        loss_c_all = self.get_loss(outputs['y_concept_all'],rshft,maskshft)
        # next
        loss_q_next = self.get_loss(outputs['y_question_next'],rshft,maskshft)#question level loss
        loss_c_next = self.get_loss(outputs['y_concept_next'],rshft,maskshft)#kc level loss
        # over all
        loss_kt = self.get_loss(outputs['y'],rshft,maskshft)

        if self.output_mode == "an_irt":
            loss = loss_kt + \
                   self.loss_q_all_lambda * loss_q_all + \
                   self.loss_c_all_lambda * loss_c_all + \
                   self.loss_c_next_lambda * loss_c_next
        else:
            loss = loss_kt + \
                   self.loss_q_all_lambda * loss_q_all + \
                   self.loss_c_all_lambda * loss_c_all + \
                   self.loss_c_next_lambda * loss_c_next + \
                   self.loss_q_next_lambda * loss_q_next

        return {
            'loss_main': loss
        }



    def get_outputs(self, emb_qc_shift, h, data, add_name="", modeltpl_type='question'):
        outputs = {}

        if modeltpl_type == 'question':
            h_next = torch.cat([emb_qc_shift, h], axis=-1)
            y_question_next = torch.sigmoid(self.out_question_next(h_next))
            y_question_all = torch.sigmoid(self.out_question_all(h))
            outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
            outputs["y_question_all" + add_name] = (y_question_all * F.one_hot(data.long(), self.n_item)).sum(-1)
        else:
            h_next = torch.cat([emb_qc_shift, h], axis=-1)
            y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
            # all predict
            y_concept_all = torch.sigmoid(self.out_concept_all(h))
            outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(y_concept_next, data)
            outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(y_concept_all, data)

        return outputs

    def get_avg_fusion_concepts(self, y_concept, cshft):
        max_num_concept = cshft[0].shape[-1]
        # concept_mask = torch.where(cshft.long()==-1, False, True)
        concept_index = F.one_hot(cshft[0], self.n_cpt)
        # concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0), self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,max_num_concept,1)*concept_index).sum(-1)
        concept_sum = concept_sum*cshft[1]#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(cshft[1].sum(-1)!=0,cshft[1].sum(-1),1)
        return y_concept

    def get_loss(self, ys,rshft,sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        loss = F.binary_cross_entropy(
            input=y_pred.double(), target=y_true.double()
        )
        return loss

class MLP(nn.Module):
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))