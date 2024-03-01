import torch

from ..pad_utils import splitter, padder, padder_list
import numpy as np
from types import SimpleNamespace
def lens2mask(lens, max_len):
        if type(lens) is int:
                masks = torch.arange(0,max_len) < lens
        else:
                masks = torch.arange(0,max_len).expand(*lens.shape, max_len) < lens.unsqueeze(-1)
        return masks.int()

def unfold_mapper(x, window_size):
        new = {}
        idx = None
        for k, v in x.items():
                if 'unfold' in k:
                        seq, lens, idx = padder(
                            v,  out_maxlen=window_size
                        )

                        new[k] =  seq
        if idx:
                stu_id = x['stu_id'][idx]
        else:
                stu_id = x['stu_id']

        ret_dict = {
            'stu_id': stu_id,
        }
        new.update(ret_dict)
        return new

def features_to_tensors(entry):
        int_features = ['lens_seq',
                        'kc_seq_lens',
                        'exer_seq',
                        'kc_seq_padding']

        float_features = ['label_seq']


        k2dtype = {k:int for k in int_features}
        k2dtype.update({k:float for k in float_features})
        entry.update({k:torch.tensor(v, dtype=k2dtype[k]) 
                     for k, v in entry.items() if k in k2dtype})


def map_yamld(entry, meta):
        """
        --naming convention
        a sequence ends with _seq
        a mask starts with mask_
        a sequence lengths start lens_
        a flattened sequence that follows one 
                kc at each item starts with flat_
        if _seq didn't mention kc or exer, it's exer
        """
        features_to_tensors(entry)
        meta = SimpleNamespace(**meta)
        entry = SimpleNamespace(**entry)

        window_size = meta.max_exer_window_size
        entry.exer_seq_mask = lens2mask(entry.lens_seq, window_size)
        entry.kc_seq = meta.kc_seq_padding[entry.exer_seq,:]

        #TODO optimize meta mask calculations  max()
        #max_kcs_per_exer = meta.kc_seq_lens.max()
        #meta.kc_seq_mask = lens2mask(meta.kc_seq_lens, max_kcs_per_exer)
        entry.kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*meta.kc_seq_mask[entry.exer_seq,:]

        #maksure to zero extras using mask, not sure if needed
        #TODO test when 0 masked get assigned Question id 0
        entry.exer_seq = entry.exer_seq_mask*entry.exer_seq
        entry.label_seq = entry.exer_seq_mask*entry.label_seq
        entry.kc_seq = entry.exer_seq_mask.unsqueeze(-1)*entry.kc_seq
        entry.kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*entry.kc_seq_mask

        return entry.__dict__
        


def map_yamld_unfold(entry, meta, is_attention=False):
        """
        --naming convention
        a sequence ends with _seq
        a mask starts with mask_
        a sequence lengths start lens_
        a flattened sequence that follows one 
                kc at each item starts with flat_
        if _seq didn't mention kc or exer, it's exer
        """
        features_to_tensors(entry)
        meta = SimpleNamespace(**meta)
        entry = SimpleNamespace(**entry)

        window_size = meta.max_exer_window_size
        entry.exer_seq_mask = lens2mask(entry.lens_seq, window_size)
        entry.kc_seq = meta.kc_seq_padding[entry.exer_seq,:]
        #max_kcs_per_exer = meta.max_kcs_per_exerc_seq_lens.max()
        #meta.kc_seq_mask = lens2mask(meta.kc_seq_lens, max_kcs_per_exer)
        entry.kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*meta.kc_seq_mask[entry.exer_seq,:]


        #question level vectors
        kc_seq = entry.kc_seq
        kc_seq_mask = entry.kc_seq_mask
        label_seq = entry.label_seq
        label_seq = label_seq.unsqueeze(-1)*kc_seq_mask
        exer_seq = entry.exer_seq
        exer_seq = exer_seq.unsqueeze(-1)*kc_seq_mask
        
        #maksure to zero extras using mask, not sure if needed
        #TODO test when 0 masked get assigned Question id 0
        exer_seq = entry.exer_seq_mask.unsqueeze(-1)*exer_seq
        kc_seq = entry.exer_seq_mask.unsqueeze(-1)*kc_seq
        kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*kc_seq_mask
        

        out = kc_seq[kc_seq_mask == 1]
        out_mask = torch.ones_like(out) 
        label_unfold_seq = label_seq[kc_seq_mask == 1]
        exer_unfold_seq = exer_seq[kc_seq_mask == 1]
        entry.kc_unfold_seq = out
        entry.unfold_seq_mask = out_mask
        entry.label_unfold_seq = label_unfold_seq
        entry.exer_unfold_seq = exer_unfold_seq

        if is_attention:
                #generate attention masks
                tmp = out_mask.unsqueeze(-1)*meta.kc_seq_mask[exer_unfold_seq,:]
                tmp = tmp.int()
                tmp = tmp[..., torch.arange(tmp.shape[-1]-1,-1,-1)]  #flipping kc sequences

                unfold_seq_len, max_num_kcs  = exer_unfold_seq.shape[0], kc_seq.shape[-1]
                tmp_len =  max_num_kcs-1 + unfold_seq_len
                attention_mask = torch.zeros(unfold_seq_len, tmp_len).int()
                idx = torch.arange(unfold_seq_len).unsqueeze(1) + torch.arange(max_num_kcs).unsqueeze(0)
                attention_mask[torch.arange(unfold_seq_len).unsqueeze(1), idx] = tmp
                attention_mask = attention_mask[:,max_num_kcs-1: ]
                entry.attention_mask = attention_mask
        
        
        #add unique names to prevent mixing them with other model implementations
        #tmp = entry.__dict__
        
        #tmp = {f"ktbench_{key}": value for key, value in tmp.items()}
        return entry.__dict__


def map_yamld_unfold(entry, meta, is_hide_label=False, is_attention=False):
        """
        --naming convention
        a sequence ends with _seq
        a mask starts with mask_
        a sequence lengths start lens_
        a flattened sequence that follows one 
                kc at each item starts with flat_
        if _seq didn't mention kc or exer, it's exer
        """
        features_to_tensors(entry)
        meta = SimpleNamespace(**meta)
        entry = SimpleNamespace(**entry)

        window_size = meta.max_exer_window_size
        entry.exer_seq_mask = lens2mask(entry.lens_seq, window_size)
        entry.kc_seq = meta.kc_seq_padding[entry.exer_seq,:]
        #max_kcs_per_exer = meta.max_kcs_per_exerc_seq_lens.max()
        #meta.kc_seq_mask = lens2mask(meta.kc_seq_lens, max_kcs_per_exer)
        entry.kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*meta.kc_seq_mask[entry.exer_seq,:]


        #question level vectors
        kc_seq = entry.kc_seq
        kc_seq_mask = entry.kc_seq_mask
        label_seq = entry.label_seq
        label_seq = label_seq.unsqueeze(-1)*kc_seq_mask
        exer_seq = entry.exer_seq
        exer_seq = exer_seq.unsqueeze(-1)*kc_seq_mask
        
        #maksure to zero extras using mask, not sure if needed
        #TODO test when 0 masked get assigned Question id 0
        exer_seq = entry.exer_seq_mask.unsqueeze(-1)*exer_seq
        kc_seq = entry.exer_seq_mask.unsqueeze(-1)*kc_seq
        kc_seq_mask = entry.exer_seq_mask.unsqueeze(-1)*kc_seq_mask
        

        out = kc_seq[kc_seq_mask == 1]
        out_mask = torch.ones_like(out) 
        label_unfold_seq = label_seq[kc_seq_mask == 1]
        exer_unfold_seq = exer_seq[kc_seq_mask == 1]
        entry.kc_unfold_seq = out
        entry.unfold_seq_mask = out_mask
        entry.label_unfold_seq = label_unfold_seq
        entry.exer_unfold_seq = exer_unfold_seq
        
        if is_hide_label:
                tmp = kc_seq_mask.clone()
                tmp[...,:-1][tmp[...,1:]==1] = 2
                double_mask = tmp[tmp>0]
                entry.masked_label_unfold_seq = entry.label_unfold_seq.clone()
                entry.masked_label_unfold_seq[double_mask==2] = 2

        if is_attention:
                #generate attention masks
                tmp = out_mask.unsqueeze(-1)*meta.kc_seq_mask[exer_unfold_seq,:]
                tmp = tmp.int()
                tmp = tmp[..., torch.arange(tmp.shape[-1]-1,-1,-1)]  #flipping kc sequences

                unfold_seq_len, max_num_kcs  = exer_unfold_seq.shape[0], kc_seq.shape[-1]
                tmp_len =  max_num_kcs-1 + unfold_seq_len
                attention_mask = torch.zeros(unfold_seq_len, tmp_len).int()
                idx = torch.arange(unfold_seq_len).unsqueeze(1) + torch.arange(max_num_kcs).unsqueeze(0)
                attention_mask[torch.arange(unfold_seq_len).unsqueeze(1), idx] = tmp
                attention_mask = attention_mask[:,max_num_kcs-1: ]
                entry.attention_mask = attention_mask
        
        
        #add unique names to prevent mixing them with other model implementations
        #tmp = entry.__dict__
        
        #tmp = {f"ktbench_{key}": value for key, value in tmp.items()}
        return entry.__dict__