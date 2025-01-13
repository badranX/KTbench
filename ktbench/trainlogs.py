from datetime import datetime
import os
from . import yamlx
from pathlib import Path
import torch
import json
from pathlib import Path
import re
import time


KTBENCH_FOLDER = ".ktbench"
ENV_FOLDER = "KTBENCH_DIR"
HUMAN2ABRV = {
    'assist2009': 'AS09',
    'corr2_assist2009': 'Synthetic',
    'duolingo2018_es_en': 'DU18',
    'riiid2020': 'RI20',
    'algebra2005': 'AL05'
}

class LogsHandler:
    def __init__(self, config, checkpoint_parent_folder=None):
        env_bench_folder = os.environ.get(ENV_FOLDER)
        if env_bench_folder:
            self.checkpoint_parent_folder = Path(env_bench_folder)
        elif checkpoint_parent_folder:
                self.checkpoint_parent_folder = checkpoint_parent_folder
        else:
            current_directory = Path.cwd()
            self.checkpoint_parent_folder  = current_directory / KTBENCH_FOLDER
            if not self.checkpoint_parent_folder.exists():
                self.checkpoint_parent_folder.mkdir()
        self.cfg = config
        self.datasetname = config.dataset_name
        self.windowsize = config.window_size
        self.dataset_window_folder = self.checkpoint_parent_folder / f"{self.datasetname}_{self.windowsize}"
        
    def train_starts(self, model_name, cfg, traincfg):
        self.timestamp = datetime.now().strftime("%Ss%Mm%Hh-%dD%mM%YY")
        append = getattr(self.cfg, 'append2logdir', '')
        self.current_checkpoint_folder = self.dataset_window_folder/(model_name + append)/self.timestamp

        self.current_checkpoint_folder.mkdir(parents=True, exist_ok=True)
        #save training parameters
        from numbers import Number
        def vdir(obj):
            return {x: getattr(obj, x) for x in dir(obj) if not x.startswith('__')}
        
        def imp(obj):
            return {k: str(v) for k, v in vdir(obj).items()}

        yamlx.write_metadata(self.current_checkpoint_folder/'traincfg.yaml', imp(traincfg))
        yamlx.write_metadata(self.current_checkpoint_folder/'cfg.yaml', imp(cfg))

    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy):
        checkpoint_filename = self.current_checkpoint_folder / "checkpoint.pth"

        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.cfg,
            'prm': model.prm,
        }, checkpoint_filename)

        # Remove the previous checkpoint folder
        if len(list(self.dataset_window_folder.glob("*"))) > 1:
            previous_checkpoint_folder = sorted(self.dataset_window_folder.glob("*"))[0]
            previous_checkpoint_folder.rmdir()

    def load_checkpoint(self, ModelClass, optimizer):
        latest_checkpoint_folder = sorted(self.dataset_window_folder.glob("*"))[-1]
        latest_checkpoint_path = latest_checkpoint_folder / "checkpoint.pth"

        # Load model and optimizer state
        checkpoint = torch.load(latest_checkpoint_path)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint['epoch']

    def load_best_model(self, device, ModelClass, kfold):
        best_model_filename = self.current_checkpoint_folder/f"best_model_fold_{kfold}.pth"

        # Load best model state
        checkpoint = torch.load(best_model_filename)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])

        return model.to(device)

    @staticmethod
    def static_load_best_model(checkpoint_folder, ModelClass, kfold, device=None):
        checkpoint_folder = Path(checkpoint_folder)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model_filename = checkpoint_folder/f"best_model_fold_{kfold}.pth"

        # Load best model state
        checkpoint = torch.load(best_model_filename)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])

        return model.to(device)

    def save_best_model(self, model, best_epoch, kfold):
        best_model_filename = self.current_checkpoint_folder/f"best_model_fold_{kfold}.pth"

        # Save best model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': best_epoch,
            'config': model.cfg,
            'prm': model.prm
        }, best_model_filename)

from pathlib import Path
import re


def simple_read_tests(directory_path, latex=True, full=False):
    def extract_timestamp(folder_name):
        try:
            val = time.strptime(folder_name, '%Ss%Mm%Hh-%dD%mM%YY')
            return time.mktime(val)
        except:
            return None

    timestamp_pattern = re.compile(r'\d{2}s\d{2}m\d{2}h-\d{2}D\d{2}M\d{4}Y')

    directory_path = Path(directory_path)

    ds_groups = set([dsdir.name.split('_')[-1] for dsdir in directory_path.iterdir() if dsdir.is_dir()])
    print("tables: ", ds_groups)
    for ds_group in ds_groups:
        timestamp_folders = {}
        meta_data = dict()
        for dsdir in directory_path.iterdir():
            if dsdir.is_dir():
                if not dsdir.name.endswith(ds_group):
                    continue
                #dataset dir
                for modeldir in dsdir.iterdir():
                    if not modeldir.is_dir():
                        continue
                    tmp = sorted([timedir for timedir in modeldir.iterdir() if timedir.is_dir() if extract_timestamp(timedir.name)],
                                                              key=lambda x: extract_timestamp(x.name),
                                                              reverse=False)

                    timestamp_folders[(dsdir.name, modeldir.name)] = tmp
                    meta_data[(dsdir.name, modeldir.name)] = [(x.name, yamlx.read_dataframe(x/'test.yaml')) for x in tmp if (x/'test.yaml').exists()]


        # df = list(meta_data.values())[0][0][1]
        # print(df)
        # mean_sem_df = df.apply(lambda x: f"{x.mean():.4f} ± {x.sem():.4f}")

        # print(mean_sem_df)
        import pandas as pd
        #out_df = pd.DataFrame()
        columns = list(set([entry[0][0] for entry in meta_data.items()]))
        index = list(set([entry[0][1] for entry in meta_data.items()]))
        mean_df = pd.DataFrame([[-111. for i in range(len(columns))] for _ in range(len(index))], index=index, columns=columns)
        sem_df = pd.DataFrame([[-111. for i in range(len(columns))] for _ in range(len(index))], index=index, columns=columns)
        for model_data, experiments in meta_data.items():
            for name, exp_df in experiments:
                if 'kfold' not in exp_df.columns or 'auc' not in exp_df.columns:
                    print("ERROR: bad df...")
                    print(model_data)
                    print(name)
                    print(exp_df)
                    print("END ERROR: bad df...")
                    print("-------")
                    continue
                min_fold =  exp_df['kfold'].min()
                max_fold =  exp_df['kfold'].max()
                if 'auc' in exp_df.columns:
                    exp_df = exp_df[['auc']]#,] 'acc']]
                else:
                    print(model_data)
                    print(name)
                    print('!!!!!!has no auc!!!!!!!')
                    continue

                #mean_sem_df = exp_df.apply(lambda x: f"{x.mean():.4f} ± {x.sem():.4f}").item()
                mean_item = exp_df.apply(lambda x: x.mean()).item()
                sem_item = exp_df.apply(lambda x: x.sem()).item()
                if min_fold == 1 and max_fold == 5:
                    mean_df.loc[model_data[1], model_data[0]] = mean_item
                    sem_df.loc[model_data[1], model_data[0]] = sem_item

                if min_fold != 1 or max_fold != 5:
                    print("min fold: ", min_fold)
                    print("max fold: ", max_fold)
                    print(model_data)
                    print(name)
                    print("len df: ", len(exp_df))
                    print("mean: ", mean_item )
                    print("sem: ", sem_item )
                    print("-------")

        print(sem_df.columns)

        if True:
            column_map = ['ASSISTments09', 'CorrAS09', 'Algebra05', 'Riiid20',  'Duoalingo2018']
            data = [[rf'${round(x,4):.4f}\pm{round(sem,4):.4f}$' for x, sem in zip(row1[1], row2[1])] 
            #data = [[x for x, sem in zip(row1, row2)] 
                for row1, row2 in zip(mean_df.iterrows(), sem_df.iterrows()) ]

            latex_ready_df = pd.DataFrame(data, 
                index=mean_df.index, columns=mean_df.columns)

            idxmax = mean_df.idxmax()
            for idx in idxmax.index:
                item = latex_ready_df.loc[idxmax[idx], idx]
                latex_ready_df.loc[idxmax[idx], idx] = rf'\bm{{{str(item)}}}'
        
            latex_ready_df.index = latex_ready_df.index.map(lambda x: rf'\textbf{{{x}}}')
            latex_ready_df.columns = latex_ready_df.columns.map(lambda x: 
                [o for o in column_map if x.lower().startswith(o[:3].lower())][0])
            #reorder columns
            latex_ready_df = latex_ready_df[column_map]
            latex_ready_df.to_pickle(directory_path.name + '.latex.pickle')
            open('latex_output.tex', 'w').write(latex_ready_df.to_latex())
        mean_df.to_pickle(directory_path.name + ".mean.pickle")
        sem_df.to_pickle(directory_path.name + ".sem.pickle")


def read_tests(directory_path, latex=True, full=False):
    def extract_timestamp(folder_name):
        try:
            val = time.strptime(folder_name, '%Ss%Mm%Hh-%dD%mM%YY')
            return time.mktime(val)
        except:
            return None

    timestamp_pattern = re.compile(r'\d{2}s\d{2}m\d{2}h-\d{2}D\d{2}M\d{4}Y')

    directory_path = Path(directory_path)

    ds_groups = set([dsdir.name.split('_')[-1] for dsdir in directory_path.iterdir() if dsdir.is_dir()])
    print("tables: ", ds_groups)
    for ds_group in ds_groups:
        timestamp_folders = {}
        meta_data = {}
        for dsdir in directory_path.iterdir():
            if dsdir.is_dir():
                if not dsdir.name.endswith(ds_group):
                    continue
                #dataset dir
                for modeldir in dsdir.iterdir():
                    if not modeldir.is_dir():
                        continue
                    tmp = sorted([timedir for timedir in modeldir.iterdir() if timedir.is_dir() if extract_timestamp(timedir.name)],
                                                              key=lambda x: extract_timestamp(x.name),
                                                              reverse=False)

                    timestamp_folders[(dsdir.name, modeldir.name)] = tmp
                    meta_data[(dsdir.name, modeldir.name)] = [(x.name, yamlx.read_dataframe(x/'test.yaml')) for x in tmp if (x/'test.yaml').exists()]
                                                            
        if full:
            for modeldir, timel in timestamp_folders.items():
                print('### ', modeldir)
                for logdir in timel:
                    print('##### ', logdir.name)
                    testfile = logdir/'test.yaml'
                    if testfile.exists():
                        print(open(testfile, 'r').read())
        else:
            import pandas as pd
            out_df = pd.DataFrame()
            benchtable = {}
            listoflists = []
            columns = ['dataset', 'model', 'auc', 'acc']
            multicolumns = set(['model'])
            for k, timel in meta_data.items():
                if not latex:
                    print('### ', k)
                dataset_name =  k[0].replace('_','-')
                model_name =  k[1].replace('_','-')
                if model_name.lower().startswith('ignore') or dataset_name.lower().startswith('ignore'):
                    continue
                row = [dataset_name, model_name]
                bench = benchtable.get(model_name, {'model': model_name})
                for traintime, df in timel:
                    if not latex:
                        print('##### ', traintime)
                        print(df.mean())
                    results = df.mean()
                    errs= df.sem()
                    auc = results['auc']
                    stderr = round(errs['auc'], 4)

                    auc = '$' + str(auc) + '\pm' + str(stderr) + '$'
                    acc = results['auc']
                    row.extend([results['auc'], results['acc']])
                    bench[(dataset_name , ' auc')] = auc
                    bench[(dataset_name , ' acc')] = acc
                    if True:
                        #only report recent logs
                        break
                
                benchtable[model_name] = bench

                listoflists.append(row)
            #df = pd.DataFrame(listoflists)
            #print(df.to_latex())
            #print('---bench---table---')
            #print()
            masked = set([v for v in benchtable.keys() if v.lower().startswith('mask')])
            nomasked = set(benchtable.keys()) - masked
            newrows = []
            for k in nomasked:
                newrows.append(k)
                similar = [s for s in masked if k.lower() in s.lower()]
                if similar:
                    mins = sorted(similar, key=lambda x: len(x))#[0]
                    for min in mins:
                        newrows.append(min)
                        masked.remove(min)
            newrows = newrows + list(masked)
            newrows = [benchtable[k] for k in newrows]
                
            df = pd.DataFrame(newrows)
            
            df.set_index('model', inplace=True)
            df.index.name = 'Model'
            if False:#+acc
                # Set the 'Model' column as the index
                
                multicolumns.remove('model')
                df.columns = pd.MultiIndex.from_tuples(df.columns)


                # Convert DataFrame to LaTeX table format
                latex = df.to_latex(
                        index=True,
                        escape=False,
                        sparsify=True,
                        multirow=True,
                        multicolumn=True,
                        multicolumn_format='c',
                        #position='p',
                        position='H',
                        bold_rows=True
                    )
                print(latex)
            else:
                df = df[[x for x in df.columns if x[-1].strip() == 'auc']]
                df.columns = list(map(lambda x: '_'.join(x[0].split('-')[:-1]),
                                       df.columns))
                df.columns = [HUMAN2ABRV.get(k, k).replace('_', '-') for k in df.columns]
                print('####dataset columns#####')
                print()
                is_tmp = False
                is_assist = False
                is_dual = False
                is_bench=False
                if is_assist and "AS09" in df.columns and "Synthetic" in df.columns:
                    df = df[["AS09", "Synthetic", "DU18"]]
                    is_tmp =True
                if is_dual and "AS09" in df.columns and "Synthetic" in df.columns:
                    df = df[["DU18"]]
                    is_tmp =True
                if is_bench:
                    try:
                        df = df[["AS09", "AL05", "RI20"]]
                    except:
                        pass
                 
                latex = df.to_latex(
                        index=True,
                        escape=False,
                        #sparsify=True,
                        #position='p',
                        bold_rows=True
                    )
                print(latex)
                print('###TRANSPOSED####')
                print()
                df  = df.T 
                if (is_assist or is_dual) and is_tmp:
                    try:
                        df = df[["DKT", 
                                 "AKT", 
                                 "SelfTeachDKT",
                                 "MaskedDKT-seperate-qa-False",
                                 "ExtraMaskAKT",
                                 "MaskedAKT",
                                 "FuseDKT",
                                 ]]
                    except Exception as e:
                        print('err: ', df.columns)
                        print()
                if is_bench:
                    try:
                        df = df[["DKT", 
                                 "AKT", 
                                 "SelfTeachDKT",
                                 "MaskedDKT-seperate-qa-False",
                                 "ExtraMaskAKT",
                                 "MaskedAKT",
                                "DKVMN",
                                "DeepIRT",
                                "QIKT"
                                 ]]
                    except Exception as e:
                        print('err: ', df.columns)
                is_tmp = False 
                is_assist= False
                is_bench =False
                latex = df.to_latex(
                        index=True,
                        escape=False,
                        #sparsify=True,
                        #position='p',
                        bold_rows=True
                    )
                print(latex)

                #latex = df.to_latex()
            # Print LaTeX table


    return timestamp_folders


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Simple Args Parser")

    parser.add_argument("directory", nargs="?", default=Path.cwd(), type=Path,
                        help="The directory to process (default: current working directory)")

    parser.add_argument("--full", action="store_true", help="Include full processing")
    parser.add_argument("--latex", action="store_true", help="Include full processing")

    args =  parser.parse_args()

    simple_read_tests(args.directory, args.latex, args.full)
