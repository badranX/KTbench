# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import pandas as pd
import os
import argparse
import random
from pathlib import Path
from . import yamlx


def process(datafolder="original_dataset", contents_dir=None, outputfolder="middata", encoding="utf-8"):
    #EdNet Nan is -1 for most fields
    # pd.set_option("mode.chained_assignment", None)  # ingore warning
    datafolder = Path(datafolder)
    contents_dir = Path(contents_dir)
    outputfolder = Path(outputfolder)

    pd.set_option("mode.chained_assignment", None)  # ignore warning
    pd.set_option("mode.chained_assignment", None)  # ignore warning
    # 读取数据集，并展示相关数据
    # 数据集包括784,309个学生，每个学生有一个交互的csv文件，这里我们进行抽样其中的5000个
    # 抽样其中的5000个
    all_files = os.listdir(datafolder)
    all_files = filter( lambda x: x != 'contents', all_files)
    random.seed(2)
    files = random.sample(list(all_files), 5000)  # 采样个数5000个
    meta = {}
    meta['used_files'] = files
    all_data = []
    for file_name in files:
        data = pd.read_csv(datafolder / file_name, encoding=encoding)
        data['stu_id'] = int(file_name[:-4][1:])
        # 先把每个用户的数据暂存到列表[]中， 后一次性转化为DataFrame
        all_data.append(data)
    df = pd.concat(all_data)
    # 确定order字段
    df = df.sort_values(by='timestamp', ascending=True)
    df['order'] = range(len(df))

    # 读取question.csv,判断用户作答情况，是否正确
    question = pd.read_csv(contents_dir/'questions.csv')

    question['tags'] = question['tags'].apply(lambda x: x.split(';')).apply(lambda x: list(map(int, x)))
    #remove exercises with no KCs
    drop_exer = question[question['tags'].apply(lambda x: x[0]<0)]['question_id']
    question = question.drop(drop_exer.index)

    #exer_cat = pd.Categorical(inter['question_id'])
    #df['question_id'] = df['question_id'].apply(lambda x: exer_cat.categories.get_loc(x))


    inter = df.merge(question, sort=False, how='left')
    inter = inter.dropna(subset=["stu_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    inter = inter.drop(inter[inter['question_id'].isin(drop_exer)].index)
    inter['label'] = (inter['correct_answer'] == inter['user_answer']).apply(int)
    inter['exer_id'] = inter['question_id']

    feature2type = {
        'stu_id': 'token', 'exer_id': 'token', 'label': 'float',
                         'start_timestamp': 'float', 'cost_time': 'float',
                         'order_id': 'token', 
                         'kc_seq': 'token_seq'
    }
    # 交互信息
    df_inter = inter \
        .reindex(columns=['stu_id', 'exer_id', 'label', 'timestamp', 'elapsed_time', 'order']) \
        .rename(columns={'stu_id': 'stu_id', 'exer_id': 'exer_id', 'label': 'label',
                         'timestamp': 'start_timestamp', 'elapsed_time': 'cost_time',
                         'order': 'order_id'})

    # 处理用户信息
    df_stu = df_inter['stu_id'].copy().unique()

    # 处理习题信息
    exer = question.copy()
    #exer['exer_id'] = pd.Categorical(exer['question_id']).codes
    # 存储所有的cpt，即知识点id
    kcs = set()
    for kc in exer['tags']:
        kcs.update(kc)
    df_exer = exer.reindex(columns=['question_id', 'tags']).rename(columns={
        'question_id': 'exer_id', 'tags': 'kc_seq'
    })
    # 此处将数据保存到`cfg.MIDDATA_PATH`中
    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    #df_user.attrs['feature2type'] = feature2type
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamlx.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    #yamlx.write_dataframe(f"{outputfolder}/stu.yaml", df_user)
    yamlx.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)
    pd.set_option("mode.chained_assignment", "warn")  # ignore warning
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', nargs='?', type=str, default='KT1', help='The target directory (default: ./KT1)')
    parser.add_argument('content_dir', nargs='?', type=str, default='contents', help='The contents directory (default: ./contents)')
    args = parser.parse_args()
    process(datafolder=args.directory, contents_dir=args.content_dir)
