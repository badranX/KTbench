from torch.utils.data import DataLoader
from noleak.trainlogs import LogsHandler
import pandas as pd
import torch

from collections import Counter
from noleak.model.dkt.dkt import DKT
from noleak.train import Trainer
from noleak.datapipeline.pipeline import Pipeline
from dataclasses import dataclass


def init_datapipeline(cfg):
    pipline = Pipeline(cfg)
    pipline.start(gen=None, from_middata=False)
    return pipline
    
def fit(cfg, traincfg):
    trainer = Trainer(traincfg, cfg)
    trainer.start()

if __name__ == '__main__':

    @dataclass
    class Cfg:
        #dataset_name = "AKT_assist2017"
        dataset_name = "assist2009"
        window_size: int = 100
        is_unfold = True
        is_unfold_fixed_window = False
        #eval_method = Trainer.EVAL_UNFOLD_KC_LEVEL
        eval_method = Trainer.EVAL_UNFOLD_REDUCE
        model_cls = DKT

    cfg = Cfg()
    cfg.logs = LogsHandler(cfg)
    trained_model = cfg.logs.load_best_model(cfg.model_cls)


    df = pd.read_pickle('topk_assist2009.pkl')
    from datasets import Dataset
    ds = Dataset.from_pandas(df)

    cfg.window_size =df.exer_id.apply(len).max()
    new_pipline = Pipeline(cfg)
    meta = new_pipline.get_meta()
    problem2KCs = meta['problem2KCs']
    paired = [{(x,y) for x in v for y in v if x != y}
              for v in problem2KCs if len(v) > 1]
    paired_dict = {x: 0 for v in paired for x in v}
    assert all([(y,x) in paired_dict for x, y in paired_dict.keys()])
    for v in paired:
        for x in v:
            paired_dict[x] = paired_dict[x] + 1
    single_kc = {x[0] for x in problem2KCs if len(x) ==1}
    paired_and_single =  Counter([x for x,_ in paired_dict.keys() if x in single_kc])
    print('paired and single: ', paired_and_single)

    print('max pairs: ', sorted(paired_dict.items(), reverse=True, key=lambda x: x[-1])[0:10])
    ds = new_pipline.process_dataset(ds)
    ds = ds.rename_columns(DKT.MODEL_FEATURE_MAP)
    input = {k: v.unsqueeze(0) for k, v in ds[0].items()}
    kc_seq = input['ktbench_kc_seq'].squeeze(0)
    kc_seq_mask = input['ktbench_kc_seq_mask'].squeeze(0)
    res =  trained_model.ktbench_trace(**input)
    res = res.squeeze(0)
    
    observations = res.reshape(-1, res.shape[-1])
    cor = torch.corrcoef(observations.T)
    tmp = torch.abs(cor)
    tmp[torch.arange(cor.shape[0]), torch.arange(cor.shape[0])] = -10
    print('max correlation: ', tmp.max())
    argmax = tmp.flatten().argmax().item()
    vals, indx = torch.topk(tmp.flatten(), 100)    
    count = 0
    for max, argmax in zip(vals, indx):
        axis1 = (argmax%tmp.shape[-1]).item()
        axis0 = (argmax//tmp.shape[-1]).item()
        assert tmp[axis0, axis1].item() == max
        if (axis0, axis1) in paired_dict:
            #print(count)
            pass
        count += 1

    idx = 0 
    seen_kcs = set()
    prev_kt = torch.ones(res.shape[-1])
    prev_kc = -100
    for kcs, kcs_mask in zip(kc_seq, kc_seq_mask):
        for kc, mask in zip(kcs, kcs_mask):
            kc = kc.item()
            if mask == 0:
                break
            kt = res[idx]
            boolidx = torch.nonzero(torch.abs(kt - prev_kt)> 0.9).flatten()
            prev_kt = kt
            for x in boolidx:
                x = x.item()
                if x not in seen_kcs and x != kc:
                    pair = (kc, x)
                    if pair in paired_dict:
                        if x in paired_and_single:
                            print('paired_and_single: ', x)
                        else:
                            print('flipped: ', pair, ':', paired_dict[pair])
                    else:
                        print('single: ', kc, 'flipped ', x)
            seen_kcs.add(kc)
            idx += 1