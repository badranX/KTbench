# -----------------------------------------------------------------------------
# Description: This file was adapted from both EduStudio and pykt-toolkit (MIT License).
# -----------------------------------------------------------------------------
import gc
import pandas as pd
import argparse
from pathlib import Path
from . import yamlx

OUTPUTFOLDER = "output"

def process(datafolder="original_dataset", outputfolder="middata", encoding="latin-1", full=False):

    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'float32', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    print("loading csv.....")
    train_file =  Path(datafolder)/'train.csv'
    question_file =  Path(datafolder)/'questions.csv'

    usecols = ["user_id", "content_id", "answered_correctly", "timestamp"]
    if full: 
        df_inter = pd.read_csv(train_file, encoding_errors='ignore', usecols=usecols, dtype=dtypes, low_memory=True)#, nrows=1e6)
    else:
        df_inter = pd.read_csv(train_file, usecols=usecols, dtype=dtypes, low_memory=True, nrows=1e6)
        
    df_exer = pd.read_csv(question_file)
    print("shape of dataframe :", df_inter.shape)

    print('with bad timestamp len: ', len(df_inter))
    df_inter= df_inter[df_inter['timestamp']>0]
    print('with bad timestamp drop len: ', len(df_inter))

    df_inter= df_inter[["user_id", "content_id", "answered_correctly", "timestamp"]]
    df_inter = df_inter[df_inter["answered_correctly"] > -0.1]
    print("shape after exlusion:", df_inter.shape)


    feature2type = {
    'stu_id': 'token', 'exer_id': 'token', 'label': 'float', 'start_timestamp': 'float', 'cost_time': 'float',
                        'order_id': 'token',
                         'kc_seq': 'token_seq', 'assignment_id': 'token_seq'
        
    }
    df_inter = df_inter[["user_id", "content_id", "answered_correctly", 'timestamp']]
    df_inter.dropna()
    df_inter = df_inter.rename(columns={'user_id': 'stu_id',
                                         'content_id': 'exer_id',
                                           'answered_correctly': 'label',
                                           'timestamp': 'order_id'})

    df_inter = df_inter.sort_values(
        ["order_id"], ascending=True)
    df_inter = df_inter.reset_index(drop=True)
    # # Save MidData
    df_inter.attrs['feature2type'] = feature2type
    ####EXERCISES###
    df_exer.attrs['feature2type'] = feature2type
    df_exer = df_exer[['question_id', 'tags']]
    df_exer.dropna(inplace=True)
    #df_exer['tags'] = df_exer['tags'].str.split().apply(lambda x: int(x))

    #df_exer = df_exer[df_exer['tags'].apply(len).bool()]
    df_exer = df_exer.rename(columns={'question_id': 'exer_id',
                                       'tags': 'kc_seq'})

    if not full:
        exers = df_exer['exer_id'].unique()
        df_inter = df_inter[df_inter['exer_id'].isin(exers)]
    
    print('start groupby...')
    group = df_inter\
        .groupby("stu_id")\
        .apply(list).reset_index(drop=True)

    print('done groupby...')
    del df_inter
    gc.collect()
    
    

    Path(outputfolder).mkdir(parents=True, exist_ok=True)
    group.to_csv(f"{outputfolder}/inter.csv", index=False)
    del group
    gc.collect()
    df_exer.to_csv(f"{outputfolder}/exer.csv", index=False)
    
    #yamlx.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    #yamlx.write_dataframe(f"{outputfolder}/inter.yaml", df_inter)
    #yamlx.write_dataframe(f"{outputfolder}/exer.yaml", df_exer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process middata')
    parser.add_argument('directory', type=str, default='middata', help='The target directory (default: ./middata)')
    parser.add_argument("--full", action="store_true", 
					help="full_dataset") 
    args = parser.parse_args()
    if args.full:
        print('###using full dataset##')
    process(datafolder=args.directory, full=args.full)
