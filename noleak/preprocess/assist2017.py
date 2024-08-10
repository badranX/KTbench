# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import pandas as pd
import argparse
from pathlib import Path
from . import yamlx
r"""
R2M_ASSIST_2017
#####################################
ASSIST_2017 dataset preprocess
"""


def process(datafolder="original_dataset", outputfolder="middata", encoding="utf-8"):
    # pd.set_option("mode.chained_assignment", None)  # ingore warning

    # 读取数据 并显示
    df = pd.read_csv(f"{datafolder}/anonymized_full_release_competition_dataset.csv", encoding=encoding, low_memory=False)
    # 进行映射
    df = df[['studentId','problemId','correct','startTime','timeTaken','MiddleSchoolId','InferredGender','skill','assignmentId']]

    # 处理skill，映射为skill_id
    knowledge_points = df['skill'].unique()
    knowledge_point_ids = {kp: idx for idx, kp in enumerate(knowledge_points, start=0)}
    df['skill_id'] = df['skill'].map(knowledge_point_ids)
    del df['skill']

    # 性别映射，空值保留
    gender_mapping = {'Male': 1, 'Female': 0}
    df['gender'] = df['InferredGender'].map(gender_mapping)
    del df['InferredGender']

    # 处理其他列
    def sort(data, column):
        '''将原始数据对指定列进行排序，并完成0-num-1映射'''
        if(column != 'startTime'):
            data .sort_values(column, inplace=True)
            value_mapping = {}
            new_value = 0
            for value in data[column].unique():
                value_mapping[value] = new_value
                new_value += 1
            new_column = f'new_{column}'
            data[new_column] = data[column].map(value_mapping)
            del data[column]
            
        # 单独处理 startTime，用字典映射
        else: 
            data = data.sort_values(by=['new_studentId', 'startTime'])

            user_mapping = {}
            user_count = {}

            def generate_mapping(row):
                '''生成作答记录时间编号映射的函数'''
                user_id = row['new_studentId']
                timestamp = row['startTime']
                if user_id not in user_mapping:
                    user_mapping[user_id] = {}
                    user_count[user_id] = 0
                if timestamp not in user_mapping[user_id]:
                    user_mapping[user_id][timestamp] = user_count[user_id]
                    user_count[user_id] += 1
                return user_mapping[user_id][timestamp]
            data['new_order_id'] = data.apply(generate_mapping, axis=1)
        return data
    df = sort(df,'studentId')
    df = sort(df,'assignmentId')
    df = sort(df,'problemId')
    df = sort(df,'MiddleSchoolId')
    df = sort(df,'startTime')


    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                        'class_id': 'token', 'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }
    
    # 修改列名及顺序
    df = df.rename(columns = {'new_studentId' : 'stu_id', 'new_assignmentId':'assignment_id','new_problemId' : 'exer_id',
                              'correct' : 'label','new_MiddleSchoolId':'school_id','new_order_id':'order_id',
                              'startTime':'start_timestamp', 'timeTaken':'cost_time','skill_id':'kc_seq'})
    new_column_order = ['stu_id','exer_id','label','start_timestamp','cost_time','order_id',
                        'school_id', 'gender','kc_seq','assignment_id']
    df = df.reindex(columns=new_column_order)

    #round to two decimal points
    df['cost_time'] = df['cost_time'].apply(lambda x: round(x,2))

    # df_inter 的相关处理
    df_inter = df[['stu_id','exer_id','label','start_timestamp','cost_time', 'order_id']].copy()
    df_inter.drop_duplicates(inplace=True)
    df_inter .sort_values('stu_id', inplace=True)

    # df_user 相关处理
    df_user = df[['stu_id','school_id','gender']].copy()
    df_user.drop_duplicates(inplace=True)
    df_user .sort_values('stu_id', inplace=True)

    # df_exer 相关处理

    # 处理列名
    df_exer = df[['exer_id','kc_seq','assignment_id']].copy()
    df_exer.sort_values(by='exer_id', inplace=True)
    df_exer.drop_duplicates(inplace=True)

    # 合并 kc_seq
    grouped_skills = df_exer[['exer_id','kc_seq']].copy()
    grouped_skills.drop_duplicates(inplace=True)
    grouped_skills.sort_values(by='kc_seq', inplace=True)
    grouped_skills['exer_id'] = grouped_skills['exer_id'].astype(str)
    grouped_skills['kc_seq'] = grouped_skills['kc_seq'].astype(str)
    grouped_skills  = grouped_skills.groupby('exer_id')['kc_seq'].agg(','.join).reset_index()

    # 合并 assignment_id
    grouped_assignments = df_exer[['exer_id','assignment_id']].copy()
    grouped_assignments.drop_duplicates(inplace=True)
    grouped_assignments.sort_values(by='assignment_id', inplace=True)
    grouped_assignments['exer_id'] = grouped_assignments['exer_id'].astype(str)
    grouped_assignments['assignment_id'] = grouped_assignments['assignment_id'].astype(str)
    grouped_assignments  = grouped_assignments.groupby('exer_id')['assignment_id'].agg(','.join).reset_index()

    # 合并结果
    df_exer = pd.merge(grouped_skills, grouped_assignments, on='exer_id', how='left')
    df_exer['exer_id'] = df_exer['exer_id'].astype(int)
    df_exer['kc_seq'] = df_exer['kc_seq'].str.split(',').apply(lambda x: list(map(int, x)))
    df_exer['assignment_id'] = df_exer['assignment_id'].str.split(',')
    df_exer.sort_values(by='exer_id', inplace=True)
    

    # # Save MidData
    # 
    # 此处将数据保存到`self.midpath`中

    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    df_user.attrs['feature2type'] = feature2type
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamlx.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamlx.write_dataframe(f"{outputfolder}/stu.yaml", df_user)
    yamlx.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    args = parser.parse_args()
    process(datafolder=args.directory)
