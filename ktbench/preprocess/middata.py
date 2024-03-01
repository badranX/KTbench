import numpy as np
import yamld
import argparse
import shutil
from sklearn import preprocessing
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

from ..pad_utils import padder_list, padder

REDUCE_PREDICT_KEYS = ['ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_unfold_seq_mask', 'ktbench_label_seq'] 
UNFOLD_KEYS = ['ktbench_exer_unfold_seq', 'ktbench_kc_unfold_seq', 'ktbench_unfold_seq_mask', 'ktbench_label_unfold_seq']
QUESTION_LEVEL_KEYS = ['ktbench_exer_seq', 'ktbench_kc_seq', 'ktbench_exer_seq_mask', 'ktbench_kc_seq_mask', 'ktbench_label_seq']

CAT2ORIGINAL = 'original'


def label2int(df, extras):
    df.label =  (df.label >= 0.5).astype(np.float32)
    return df, extras

def kcs_str2list(df, extras):
    df_exer = extras['exer_df']
    df_exer.kc_seq = df_exer.kc_seq.apply(lambda x: list(map(int, x.split(","))))
    return df, extras

def kcs2list(df, extras):
    df_exer = extras['exer_df']
    sample = df_exer.kc_seq[0]
    if isinstance(sample, list):
        print('[INFO] KCs are already a list')
        return df, extras
    df_exer.kc_seq = df_exer.kc_seq.apply(lambda x: x.split(","))
    return df, extras


def normalize_time(df, extras):
    timestamp = 'start_timestamp' 
    if timestamp in df:
        df_exer = extras['exer_df']
        df_exer.kc_seq = df_exer.kc_seq.apply(lambda x: list(map(int, x.split(","))))
        return df, extras
    else:
        print("[WARNING] start_timestamp not in dataframe")
    return df, extras
def _factorise_df(df, extras, already_factorized, feature2type, ignore=[]):
    #tokens
    extras[CAT2ORIGINAL] = extras.get(CAT2ORIGINAL, {})
    columns = [c for c in df.columns if feature2type[c] == 'token'
              and c != 'order_id']
    for c in columns:
        if c in ignore:
            continue
        if c in already_factorized:
            cat = already_factorized[c]
            df[c] = df[c].apply(lambda x: cat.get_loc(x))
        else:
            df[c], cat = pd.factorize(df[c])
            extras[CAT2ORIGINAL][c] = {k:v for k, v in enumerate(cat)}
            already_factorized[c] = cat

    #token_seq
    columns = [c for c in df.columns if feature2type[c] == 'token_seq'
              and c != 'order_id']
    for c in columns:
        if c in ignore:
            continue
        assert c not in already_factorized
        le = preprocessing.LabelEncoder()
        le.fit(df[c].explode())
        extras[CAT2ORIGINAL][c] = {k:v for k, v in enumerate(le.classes_)}
        df[c] = df[c].apply(lambda x: le.transform(x).tolist())
        already_factorized[c] = None

    return df, extras, already_factorized

def factorize(df, extras):
    feature2type = df.attrs['feature2type']
    df_exer = extras['exer_df']
    already_factorized = {}
    df_exer, extras, already_factorized = _factorise_df(df_exer, extras, already_factorized, feature2type)
    if 'stu_df' in extras:
        df_stu = extras['stu_df']
        df_stu, extras, already_factorized = _factorise_df(df_stu, extras, already_factorized, feature2type)
    #tokens
    df, extras, already_factorized = _factorise_df(df, extras, already_factorized, feature2type)
    return df, extras


def sort_by_orderid(df, extras):
    df = df.sort_values(by="order_id", ascending=True).reset_index(drop=True)
    return df, extras



def is_middata_ready(middata_dir):
    middata_dir = Path(middata_dir)
    exer = (middata_dir / "exer.yaml").exists()
    #stu = (middata_dir / "stu.yaml").exists()
    inter = (middata_dir / "inter.yaml").exists()
    return exer and inter

def read_middata(middata_dir="./middata"):
    if not is_middata_ready(middata_dir=middata_dir):
        raise Exception("something is wrong with middata")

    print("reading datasets from middata...")
    exer = yamld.read_dataframe(f"{middata_dir}/exer.yaml", encoding='utf-8')
    inter = yamld.read_dataframe(f"{middata_dir}/inter.yaml", encoding='utf-8')
    stu_path = f"{middata_dir}/stu.yaml"
    ret = {'inter_df': inter, 'exer_df': exer}
    if Path(stu_path).exists():
        stu = yamld.read_dataframe(stu_path, encoding='utf-8')
        ret['stu_df'] = stu
    print("done reading middata...")
    return ret


def write_processed_data(path, df, extras):
    meta = extras.get('meta', {})
    df.attrs.update(meta)
    yamld.write_dataframe(path, df)

def gen_kc_seq(df, extras):
    df_exer = extras['exer_df']
    mapping = df_exer[['exer_id', 'kc_seq']].copy()
    mapping['kc_seq'] = mapping['kc_seq'].apply(tuple)
    mapping.drop_duplicates(inplace=True)
    mapping = mapping.values.tolist()
    sorted(mapping, key=lambda x: x[0])
    kc_seq_unpadding = list(map(lambda x: x[1], mapping))
    kc_count = len(set(kc_seq_unpadding))
    exer_count = len(df_exer['exer_id'].unique())

    #assertions
    
    unique = len(set(map(tuple, kc_seq_unpadding)))
    print('unique kcs : ', unique)
    
    
    meta = extras.get('meta', {})
    extras['meta'] = meta
    
    meta['problem2KCs'] = kc_seq_unpadding
    meta['n_exer'] = exer_count
    meta['n_kc'] = kc_count

    return df, extras

def gen_kc_seq_with_padding(df, extras):
    #TODO handle split data

    df_exer = extras['exer_df']

    tmp_df_Q = df_exer.set_index('exer_id')


    #TODO counting vs largest index?
    #kc_count = len(df_exer['kc_seq'].explode().unique())
    kc_count = int(df_exer.kc_seq.explode().max()+1)
    exer_count = int(df.exer_id.max() + 1)

    
    kc_seq_unpadding = [
        (tmp_df_Q.loc[exer_id].kc_seq if exer_id in tmp_df_Q.index else []) for exer_id in range(exer_count)
    ]

    kc_seq_padding, kc_seq_lens , _ = padder_list(kc_seq_unpadding, out_maxlen=-1, dtype=int)

    #save as lists
    meta = extras.get('meta', {})
    extras['meta'] = meta
    meta['kc_seq_padding'] = kc_seq_padding
    meta['kc_seq_lens'] = kc_seq_lens
    meta['n_exer'] = exer_count
    meta['n_kc'] = kc_count
    #calculate the max_size window if unfolded cpts
    return df, extras

def groupby_student(df, extras):
    meta = extras.get('meta', {})
    extras['meta'] = meta
    df = df.groupby(df.stu_id).agg(list).reset_index()
    window_size = df['exer_id'].apply(len).max()
    meta['max_window_size'] = window_size
    return df, extras

def process_middata(middata_dir= "./middata", outpath="./middata.yaml"):
    PIPELINE = [factorize, sort_by_orderid, gen_kc_seq, groupby_student]
    print("[Debug] processing from middata")
    dfs = read_middata(middata_dir=middata_dir)
    df = dfs['inter_df']
    extras = {'meta': {}}
    extras.update(dfs)
    
    print("start processing middata...")
    for func in PIPELINE:
        print(df.columns)
        df, extras = func(df, extras)
        
    # Save original mappings
    if CAT2ORIGINAL in extras:
        savepath = Path(outpath).parent / 'out2original.yaml'
        yamld.write_metadata(savepath, extras[CAT2ORIGINAL])

    if 'start_timestamp' in df:
        print("normalize start_timestamp...")
        df['start_timestamp'] = df['start_timestamp'].apply(lambda l: [x - min(l) for x in l])
    else:
        print('[WARNING] start_timestamp not available')
    print("done processing middata.")
    print("start writing processed data...")
    df = df.drop(columns=['order_id'])
    write_processed_data(outpath, df, extras)
    print("done writing processed data.")

    return df, extras


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, nargs='?', default='./middata', help='The target directory (default: ./middata)')

    args = parser.parse_args()
    process_middata(middata_dir=args.directory)