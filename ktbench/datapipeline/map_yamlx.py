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
                for k, v in x.items():
                        if 'unfold' not in k:
                                new[k] = x[k][idx]


        x.update(new)
        return x

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

        if False:
            problem2Vals_keys = {k: v for k, v in entry.items() if k.startswith('extra_problem2')}
            for k, v in problem2Vals_keys.items():
                tmp_name = k[len('extra_problem2'):]
                meta.update({k: torch.tensor(v)})
                setattr(entry, tmp_name, meta.kc_seq_padding[entry.exer_seq,:])


def map_yamlx(entry, meta):
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
        entry.__dict__.pop('lens_seq')
        entry.kc_seq = meta.kc_seq_padding[entry.exer_seq,:]
        problem2Vals_keys = [k for k in meta.__dict__.keys() if k.startswith('extra_problem2')]
        for k in problem2Vals_keys:
            tmp_name = k[len('extra_problem2'):]
            problem2Val_map = getattr(meta, k)
            setattr(entry, tmp_name, problem2Val_map[entry.exer_seq,:])

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


def map_yamlx_unfold(entry, meta, is_mask_label=False, is_teacher_mask=False, is_attention=False):
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
        entry.__dict__.pop('lens_seq')
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
        
        if is_mask_label or is_teacher_mask:
                tmp = kc_seq_mask.clone()
                tmp[...,:-1][tmp[...,1:]==1] = 2
                double_mask = tmp[tmp>0]
                entry.masked_label_unfold_seq = entry.label_unfold_seq.clone()
                entry.masked_label_unfold_seq[double_mask==2] = 2
                if is_teacher_mask:
                        entry.teacher_unfold_seq_mask = (double_mask == 2).int()

        if is_attention:

                kc_group_seq = meta.kc_seq_mask[exer_unfold_seq,:]
                group_lens = torch.arange(0, kc_group_seq.shape[0]) - kc_group_seq.sum(-1)
                group_lens = torch.clamp(group_lens, min=0)
                attention_mask = torch.arange(kc_group_seq.shape[0]).unsqueeze(0) < group_lens.unsqueeze(-1)
                
                entry.attention_mask = attention_mask
        
        
        #add unique names to prevent mixing them with other model implementations
        #tmp = entry.__dict__
        
        #tmp = {f"ktbench_{key}": value for key, value in tmp.items()}
        return entry.__dict__


ALLINONETAG = 'allinone_ktbench_'
def map_allinone_before_batch(entry, tgt_features, is_hide_label=False):
        entry = SimpleNamespace(**entry)

        new = SimpleNamespace()

        expanded_label_seq = entry.ktbench_label_seq.unsqueeze(-1)*entry.ktbench_kc_seq_mask
        expanded_exer_seq = entry.ktbench_exer_seq.unsqueeze(-1)*entry.ktbench_kc_seq_mask
        
        new.ktbench_kc_unfold_seq = []
        new.ktbench_unfold_seq_mask = []
        new.ktbench_label_unfold_seq = []
        new.ktbench_exer_unfold_seq = []

        new.ktbench_allinone_label = []
        new.ktbench_allinone_id = []
        new.ktbench_allinone_tgt_index = []

        if is_hide_label:
                new.ktbench_masked_label_unfold_seq = []
                new.ktbench_teacher_unfold_seq_mask = []
                clone_kc_seq_mask = entry.ktbench_kc_seq_mask.clone()
                
        tgt_replicate_keys = set(tgt_features) - set(new.__dict__.keys())
        tgt_replicate_keys = tgt_replicate_keys.intersection(set(entry.__dict__.keys()))
        new.__dict__.update({k: [] for k in tgt_replicate_keys})

        for idx  in range(2, entry.ktbench_exer_seq_mask.shape[-1] + 1):
                if entry.ktbench_exer_seq_mask[idx - 1] == 0:
                        break

                allone_id = entry.ktbench_idx*entry.ktbench_label_seq.shape[-1]  + idx - 1
                end_label = entry.ktbench_label_seq[idx-1]

                out = entry.ktbench_kc_seq[:idx][entry.ktbench_kc_seq_mask[:idx] == 1]
                out_mask = torch.ones_like(out) 
                label_unfold_seq = expanded_label_seq[:idx][entry.ktbench_kc_seq_mask[:idx] == 1]
                exer_unfold_seq = expanded_exer_seq[:idx][entry.ktbench_kc_seq_mask[:idx] == 1]

                tmplen = entry.ktbench_kc_seq_mask[idx-1].sum(-1).item()
                seqlen = exer_unfold_seq.shape[-1] - tmplen
                indices = [list(range(seqlen)) + [seqlen+j] for j in range(tmplen)]
                #tgt = torch.zeros(seqlen+1)
                #tgt[-1] = 1


                if is_hide_label:
                        #original
                        tmp = clone_kc_seq_mask[:idx]
                        tmp[...,:-1][tmp[...,1:]==1] = 2
                        double_mask = tmp[tmp>0]

                        masked_label_unfold_seq = label_unfold_seq.clone()
                        masked_label_unfold_seq[double_mask==2] = 2
                        teacher_unfold_seq_mask = (double_mask == 2).int()

                        new.ktbench_masked_label_unfold_seq += list(masked_label_unfold_seq[None,indices].squeeze(0))
                        new.ktbench_teacher_unfold_seq_mask += list(teacher_unfold_seq_mask[None,indices].squeeze(0))
                        
                new.ktbench_allinone_label += [end_label]*tmplen
                new.ktbench_allinone_id += [allone_id]*tmplen
                #new.ktbench_allinone_mask += [tgt]*tmplen
                new.ktbench_allinone_tgt_index += [seqlen]*tmplen

                new.ktbench_kc_unfold_seq += list(out[None,indices].squeeze(0).tolist())
                new.ktbench_unfold_seq_mask += list(out_mask[None, indices].squeeze(0).tolist())
                new.ktbench_label_unfold_seq += list(label_unfold_seq[None,indices].squeeze(0).tolist())
                new.ktbench_exer_unfold_seq += list(exer_unfold_seq[None,indices].squeeze(0).tolist())
                new.__dict__.update({k: new.__dict__[k] + tmplen*[entry.__dict__[k].tolist()] for k in tgt_replicate_keys})
                
        return new.__dict__


def tmp_map_allinone_batch(entry):
        return {k: [item for vv in v for item in vv] for k, v in entry.items()}


def map_allinone_batch(entry):
        ret = dict()
        for k, v in entry.items():
                result = []
                extend = result.extend
                for l in v:
                        extend(l)
                ret[k] = result
        return ret
