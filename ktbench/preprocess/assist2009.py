import pandas as pd
import argparse
from pathlib import Path
import yamld

r"""
This code was originally copied from EduStudio
source: https://github.com/HFUT-LEC/EduStudio

R2M_ASSIST_0910
#####################################
ASSIST_0910 dataset preprocess
"""
OUTPUTFOLDER = "output"

def process(datafolder="original_dataset", outputfolder="middata", encoding="latin-1"):
    pd.set_option("mode.chained_assignment", None)  # ingore warning
    df = pd.read_csv(f"{datafolder}/skill_builder_data.csv", encoding=encoding, low_memory=False)

    df = df.dropna(subset=['skill_id'])

    df['class_id'] = df['school_id'].astype(str) + '_' + df['student_class_id'].astype(str)


    df = df[['user_id', 'assignment_id', 'problem_id', 'correct', 'ms_first_response', 'overlap_time', 'class_id',
             'skill_id']]

    def sort(data, column):
        '''将原始数据对指定列进行排序，并完成0-num-1映射'''

        # 非 class_id、非ms_first_response单独处理
        if (column != 'class_id'):
            data.sort_values(column, inplace=True)

        # 处理 class_id
        elif (column != 'ms_first_response'):
            data[['school_id', 'group_id']] = data['class_id'].str.split('_', expand=True)
            data['school_id'] = data['school_id'].astype(int)
            data['group_id'] = data['group_id'].astype(int)
            data.sort_values(['school_id', 'group_id'], inplace=True)

        # 映射非 ms_first_response 列
        if (column != 'ms_first_response'):
            value_mapping = {}
            new_value = 0
            for value in data[column].unique():
                value_mapping[value] = new_value
                new_value += 1

            # 映射 生成新列
            new_column = f'new_{column}'
            data[new_column] = data[column].map(value_mapping)
            del data[column]

        # 处理ms_first_response，每个同学答题顺序单独映射
        else:
            data = data.sort_values(by=['new_user_id', 'ms_first_response'])

            # 创建空字典存储同学的作答记录时间编号映射
            user_mapping = {}
            user_count = {}

            def generate_mapping(row):
                '''生成作答记录时间编号映射的函数'''
                user_id = row['new_user_id']
                timestamp = row['ms_first_response']
                if user_id not in user_mapping:
                    user_mapping[user_id] = {}
                    user_count[user_id] = 0
                if timestamp not in user_mapping[user_id]:
                    user_mapping[user_id][timestamp] = user_count[user_id]
                    user_count[user_id] += 1
                return user_mapping[user_id][timestamp]

            # 映射，并生成新列
            data['new_order_id'] = data.apply(generate_mapping, axis=1)

        return data

    # 处理各列
    df = sort(df, 'user_id')
    df = sort(df, 'assignment_id')
    df = sort(df, 'problem_id')
    df = sort(df, 'skill_id')
    df = sort(df, 'class_id')
    df = sort(df, 'ms_first_response')
    del df['school_id']
    del df['group_id']
    # print(df)
    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                        'class_id': 'token', 'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }
    # 修改列名
    df = df.rename(
        columns={'new_user_id': 'stu_id', 'new_problem_id': 'exer_id', 'correct': 'label',
                 'ms_first_response': 'start_timestamp', 'new_class_id': 'class_id',
                 'new_skill_id': 'kc_seq', 'overlap_time': 'cost_time',
                 'new_assignment_id': 'assignment_id', 'new_order_id': 'order_id'})
    # 指定列的新顺序
    new_column_order = ['stu_id', 'exer_id', 'label', 'start_timestamp', 'cost_time',
                        'order_id',
                        'class_id', 'kc_seq', 'assignment_id']
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

    # df_user 相关处理
    df_user = df[['stu_id', 'class_id']]
    df_user.drop_duplicates(inplace=True)
    df_user.sort_values('stu_id', inplace=True)
    # print(df_user)

    # df_exer 相关处理

    # 处理列名
    df_exer = df[['exer_id', 'kc_seq', 'assignment_id']]
    df_exer.sort_values(by='exer_id', inplace=True)
    df_exer.drop_duplicates(inplace=True)

    # 合并 cpt_seq
    grouped_skills = df_exer[['exer_id', 'kc_seq']]
    grouped_skills.drop_duplicates(inplace=True)
    grouped_skills.sort_values(by='exer_id', inplace=True)
    grouped_skills['exer_id'] = grouped_skills['exer_id'].astype(str)
    grouped_skills['kc_seq'] = grouped_skills['kc_seq'].astype(str)
    grouped_skills = grouped_skills.groupby('exer_id')['kc_seq'].agg(','.join).reset_index()

    # 合并 assignment_id
    grouped_assignments = df_exer[['exer_id', 'assignment_id']]
    grouped_assignments.drop_duplicates(inplace=True)
    grouped_assignments.sort_values(by='assignment_id', inplace=True)
    grouped_assignments['exer_id'] = grouped_assignments['exer_id'].astype(str)
    grouped_assignments['assignment_id'] = grouped_assignments['assignment_id'].astype(str)
    grouped_assignments = grouped_assignments.groupby('exer_id')['assignment_id'].agg(
        ','.join).reset_index()

    # 合并结果
    df_exer = pd.merge(grouped_skills, grouped_assignments, on='exer_id', how='left')
    df_exer['exer_id'] = df_exer['exer_id'].astype(int)
    df_exer['kc_seq'] = df_exer['kc_seq'].str.split(',').apply(lambda x: list(map(int, x)))
    df_exer['assignment_id'] = df_exer['assignment_id'].str.split(',')
    df_exer.sort_values(by='exer_id', inplace=True)
    # print(df_exer)

    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    df_exer.attrs['feature2type'] = feature2type
    df_user.attrs['feature2type'] = feature2type
    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamld.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamld.write_dataframe(f"{outputfolder}/stu.yaml", df_user)
    yamld.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)
    pd.set_option("mode.chained_assignment", "warn")  # igore warning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    args = parser.parse_args()
    process(datafolder=args.directory)
