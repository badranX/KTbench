# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import yamld

TIME_FORMAT =   "%Y-%m-%d %H:%M:%S.%f"

OUTPUTFOLDER = "output"
#!/usr/bin/env python
# coding=utf-8

import pandas as pd

KEYS = ["Anon Student Id", "KC(Default)", "Questions"]

def change2timestamp(t):
    return datetime.strptime(t, TIME_FORMAT).timestamp()

def format_list2str(input_list):
    return [str(x) for x in input_list]

def process(read_dir, outputfolder='middata'):
    read_dir = Path(read_dir)
    raw_data_train = pd.read_table(read_dir / "algebra_2005_2006_train.txt", encoding='utf-8')
    raw_data_test = pd.read_table(read_dir / "algebra_2005_2006_master.txt", encoding='utf-8')
    df = pd.concat([raw_data_train, raw_data_test], axis=0)
    #df["Problem Name"] = df["Problem Name"].apply(replace_text)
    #df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1)
 
    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"])
    df = df[df["Correct First Attempt"].isin([0,1])]
    
    #map
    df["KC(Default)"] = df["KC(Default)"].apply(lambda x: x.replace("~~", "_").split('_') )
    df["First Transaction Time"] = df["First Transaction Time"].apply(change2timestamp)
    df["order_id"] = df[["First Transaction Time", "index"]].values.tolist()

    df = df[["order_id", "Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"]]

    #write_txt(write_file, data)
    #df = pd.DataFrame(data)
    df = df.rename(columns= {
        'Anon Student Id': 'stu_id',
       'Questions': 'exer_id',
       'KC(Default)': 'kc_seq',
       'Correct First Attempt': 'label',
       'First Transaction Time': 'start_timestamp',
    })

    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                        'class_id': 'token', 'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }

    df_exer = df[['exer_id', 'kc_seq']]
    #kc_id is long string, this factorize it
    #test, cat_kcs = pd.factorize(df_exer.explode('kc_seq')['kc_seq'])
    #df_exer.loc[:,'kc_seq'] = df_exer['kc_seq'].apply(lambda x: list(map(cat_kcs.get_loc, x)))
    
    df_exer.loc[df_exer.astype(str).drop_duplicates().index]
    df_inter = df.drop(columns=['kc_seq'])
    
    
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    yamld.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamld.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    args = parser.parse_args()
    process(read_dir=args.directory)

