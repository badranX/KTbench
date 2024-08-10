# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import pandas as pd
import argparse
from pathlib import Path
from . import yamlx

OUTPUTFOLDER = "output"

def process(datafolder="original_dataset", outputfolder="middata", encoding="latin-1"):
    pd.set_option("mode.chained_assignment", None)  # ingore warning
    df = pd.read_csv(f"{datafolder}/skill_builder_data.csv", encoding=encoding, low_memory=False)
    df['tmp_index'] = range(len(df))

    df['order_id'] = df[['order_id','tmp_index']].values.tolist()
    df = df.dropna(subset=['skill_id'])
    df = df.dropna(subset=["user_id","problem_id", "skill_id", "correct", "order_id"])


    df = df[['user_id', 'assignment_id', 'problem_id', 'correct', 'ms_first_response', 'overlap_time', 'skill_id']]

    # print(df)
    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                         'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }
    # 修改列名
    df = df.rename(
        columns={'user_id': 'stu_id', 'problem_id': 'exer_id', 'correct': 'label',
                 'ms_first_response': 'start_timestamp', 'skill_id': 'kc_seq', 'overlap_time': 'cost_time',
                 'assignment_id': 'assignment_id', 'order_id': 'order_id'})
    # 指定列的新顺序
    new_column_order = ['stu_id', 'exer_id', 'label', 'start_timestamp', 'cost_time',
                        'order_id', 'kc_seq', 'assignment_id']
    df = df.reindex(columns=new_column_order)
    # print(df)

    #round to two decimal points
    df['cost_time'] = df['cost_time'].apply(lambda x: round(x,2))

    # df_inter 相关处理
    df_inter = df[['stu_id', 'exer_id', 'label', 'start_timestamp', 'cost_time',
                   'order_id']]
    df_inter.drop_duplicates(inplace=True)
    df_inter.sort_values('stu_id', inplace=True)
    # print(df_inter)

    # df_exer 相关处理

    # 处理列名
    df_exer = df[['exer_id', 'kc_seq', 'assignment_id']]
    df_exer.sort_values(by='exer_id', inplace=True)
    df_exer.drop_duplicates(inplace=True)

    # 合并 cpt_seq
    grouped_skills = df_exer[['exer_id', 'kc_seq']]
    grouped_skills.drop_duplicates(inplace=True)
    grouped_skills.sort_values(by='exer_id', inplace=True)
    grouped_skills['exer_id'] = grouped_skills['exer_id'].astype(int)
    grouped_skills['kc_seq'] = grouped_skills['kc_seq'].astype(int)
    grouped_skills = grouped_skills.groupby('exer_id').agg(set).reset_index()
    grouped_skills['kc_seq'] = grouped_skills['kc_seq'].apply(list)
    grouped_skills['kc_seq'] = grouped_skills['kc_seq'].apply(sorted)


    # 合并结果
    df_exer = grouped_skills
    df_exer['exer_id'] = df_exer['exer_id'].astype(int)
    print('len exers: ' , len(df_exer['exer_id'].unique()))
    #df_exer['kc_seq'] = df_exer['kc_seq'].str.split(',').apply(lambda x: list(map(lambda x:int(float(x)), x)))
    df_exer['kc_seq'] = df_exer['kc_seq'].apply(lambda x: list(map(lambda x:int(float(x)), x)))
    print('unique kc groups: ', len(df_exer['kc_seq'].apply(str).unique()))
    df_exer.sort_values(by='exer_id', inplace=True)
    # print(df_exer)

    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamlx.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamlx.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)
    pd.set_option("mode.chained_assignment", "warn")  # igore warning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    args = parser.parse_args()
    process(datafolder=args.directory)
