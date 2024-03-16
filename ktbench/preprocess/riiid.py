# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------

import pandas as pd
import argparse
from pathlib import Path
import yamld

OUTPUTFOLDER = "output"

def process(datafolder="original_dataset", outputfolder="middata", encoding="latin-1", full=False):

    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'float32', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    print("loading csv.....")
    train_file =  Path(datafolder)/'train.csv'
    question_file =  Path(datafolder)/'questions.csv'
    if full: 
        train_df = pd.read_csv(train_file, dtype=dtypes, low_memory=True)#, nrows=1e6)
    else:
        train_df = pd.read_csv(train_file, dtype=dtypes, low_memory=True, nrows=1e6)
        
    df_exer = pd.read_csv(question_file)
    print("shape of dataframe :", train_df.shape)

    print('with bad timestamp len: ', len(train_df))
    train_df= train_df[train_df['timestamp']>0]
    print('with bad timestamp drop len: ', len(train_df))

    train_df = train_df.sort_values(
        ["timestamp"], ascending=True)
    train_df = train_df.reset_index(drop=True)
    train_df= train_df[["user_id", "content_id", "answered_correctly", "timestamp"]]
    train_df = train_df[train_df["answered_correctly"] > -0.1]
    n_skills = train_df.content_id.nunique()
    print("no. of skills :", n_skills)
    print("shape after exlusion:", train_df.shape)


    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                         'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }
    df_inter = train_df[["user_id", "content_id", "answered_correctly", 'timestamp']]
    df_inter = df_inter.rename(columns={'user_id': 'stu_id',
                                         'content_id': 'exer_id',
                                           'answered_correctly': 'label',
                                           'timestamp': 'order_id'})
    df_inter.dropna()
    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    ####EXERCISES###
    df_exer.attrs['feature2type'] = feature2type
    df_exer = df_exer[['question_id', 'tags']]
    df_exer.dropna(inplace=True)
    df_exer['tags'] = df_exer['tags'].str.split()
    #df_exer = df_exer[df_exer['tags'].apply(len).bool()]
    df_exer = df_exer.rename(columns={'question_id': 'exer_id',
                                       'tags': 'kc_seq'})

    exers = df_exer['exer_id'].unique()
    df_inter = df_inter[df_inter['exer_id'].isin(exers)]

    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    yamld.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    yamld.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)
    pd.set_option("mode.chained_assignment", "warn")  # igore warning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    parser.add_argument("--full", action="store_true", 
					help="full_dataset") 
    args = parser.parse_args()
    if args.full:
        print('###using full dataset##')
    process(datafolder=args.directory, full=args.full)
