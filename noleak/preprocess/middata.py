import numpy as np
from . import yamlx
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

FEATURE2TYPE = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                         'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
    }

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
            df.loc[:, c] = df[c].apply(lambda x: cat.get_loc(x))
        else:
            df.loc[:, c], cat = pd.factorize(df[c])
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
        df.loc[:, c] = df[c].apply(lambda x: le.transform(x).tolist())
        already_factorized[c] = None

    return df, extras, already_factorized

def factorize(df, extras):
    feature2type = df.attrs['feature2type']
    df_exer = extras['exer_df']
    df_exer = df_exer.drop_duplicates(['exer_id'])
    already_factorized = {}
    df_exer, extras, already_factorized = _factorise_df(df_exer, extras, already_factorized, feature2type)
    extras['exer_df'] = df_exer
    if 'stu_df' in extras:
        df_stu = extras['stu_df']
        df_stu = df_stu.drop_duplicates(['stu_id'])
        df_stu, extras, already_factorized = _factorise_df(df_stu, extras, already_factorized, feature2type)
        extras['stu_df'] = df_stu
    #tokens
    df, extras, already_factorized = _factorise_df(df, extras, already_factorized, feature2type)
    return df, extras


def sort_by_orderid(df, extras):
    df = df.sort_values(by="order_id", ascending=True).reset_index(drop=True)
    return df, extras



def is_middata_ready(middata_dir, from_csv=False):
    append = "csv" if from_csv else "yaml"
    middata_dir = Path(middata_dir)
    exer = (middata_dir / f"exer.{append}").exists()
    #stu = (middata_dir / "stu.yaml").exists()
    inter = (middata_dir / f"inter.{append}").exists()
    return exer and inter

def read_middata(middata_dir="./middata", from_csv=False):
    if not is_middata_ready(middata_dir=middata_dir, from_csv=from_csv):
        raise Exception("something is wrong with middata")

    print("reading datasets from middata...")
    if from_csv:
       exer = pd.read_csv(f"{middata_dir}/exer.csv", encoding='utf-8', low_memory=True) 
       exer['kc_seq'] = exer['kc_seq'].str.split()
       inter = pd.read_csv(f"{middata_dir}/inter.csv", encoding='utf-8', low_memory=True) 
    else:
        exer = yamlx.read_dataframe(f"{middata_dir}/exer.yaml", encoding='utf-8')
        inter = yamlx.read_dataframe(f"{middata_dir}/inter.yaml", encoding='utf-8')
    stu_path = f"{middata_dir}/stu.yaml"
    ret = {'inter_df': inter, 'exer_df': exer}
    if Path(stu_path).exists():
        if from_csv:
           pd.read_csv(stu_path, encoding='utf-8', low_memory=True) 
        else:
            stu = yamlx.read_dataframe(stu_path, encoding='utf-8')
            ret['stu_df'] = stu
        
    print("done reading middata...")
    return ret


def write_processed_data(path, df, extras):
    meta = extras.get('meta', {})
    df.attrs.update(meta)
    yamlx.write_dataframe(path, df)

def gen_kc_seq(df, extras):
    df_exer = extras['exer_df']
    #df_exer = df_exer.drop_duplicates(['exer_id'])
    try:
        assert df_exer['exer_id'].max() + 1 == len(df_exer)
    except AssertionError as e:
        print('max :', df_exer['exer_id'].max())
        print('df len: ', len(df_exer))
        raise e

    exploded_kc_seq = df_exer['kc_seq'].explode().unique()
    try:
        assert exploded_kc_seq.max() + 1 == len(exploded_kc_seq)
    except AssertionError as e:
        print('max :', exploded_kc_seq.max() )
        print('df len: ', len(exploded_kc_seq))
        raise e
    kc_count = len(exploded_kc_seq)
    exploded_kc_seq = None

    exer_count = df_exer['exer_id'].nunique()
    stu_count = df['stu_id'].nunique()
    grouped_kc_count = df_exer['kc_seq'].apply(tuple).nunique()
    avg_kc_per_exer = df_exer['kc_seq'].apply(len).mean()

    kc_seq_unpadding = df_exer.sort_values(by='exer_id')['kc_seq'].values.tolist()

    print('unique kcs : ', grouped_kc_count)
    
    meta = extras.get('meta', {})
    extras['meta'] = meta
    
    meta['problem2KCs'] = kc_seq_unpadding
    meta['n_exer'] = exer_count
    meta['n_kc'] = kc_count
    meta['n_stu'] = stu_count
    meta['grouped_kc_count'] = grouped_kc_count
    meta['avg_kc_per_exer'] = avg_kc_per_exer

    return df, extras

def gen_kc_seq_with_padding(df, extras):
    #TODO handle split data

    df_exer = extras['exer_df']

    tmp_df_Q = df_exer.set_index('exer_id')

    exer_count = meta['n_exer']
    
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
    return df, extras

def groupby_student(df, extras):
    meta = extras.get('meta', {})
    extras['meta'] = meta
    df = df.groupby(df.stu_id).agg(list).reset_index()
    window_size = df['exer_id'].apply(len).max()
    meta['max_window_size'] = window_size
    return df, extras

def process_middata(middata_dir= "./middata", outpath="./middata.yaml", from_csv=False):
    PIPELINE = [factorize, sort_by_orderid, gen_kc_seq, groupby_student]
    print("[Debug] processing from middata")
    dfs = read_middata(middata_dir=middata_dir, from_csv=from_csv)
    for df in dfs.values():
        if not isinstance(df, pd.DataFrame):
            continue
        if 'feature2type' not in df.attrs:
            df.attrs['feature2type'] = FEATURE2TYPE
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
        yamlx.write_metadata(savepath, extras[CAT2ORIGINAL])

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

    parser.add_argument("--csv", action="store_true", 
					help="from a csv middata files") 
    args = parser.parse_args()
    process_middata(middata_dir=args.directory, from_csv=args.csv)