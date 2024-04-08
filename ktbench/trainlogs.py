from datetime import datetime
import os
import yamld
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

        yamld.write_metadata(self.current_checkpoint_folder/'traincfg.yaml', imp(traincfg))
        yamld.write_metadata(self.current_checkpoint_folder/'cfg.yaml', imp(cfg))

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
                    meta_data[(dsdir.name, modeldir.name)] = [(x.name, yamld.read_dataframe(x/'test.yaml')) for x in tmp if (x/'test.yaml').exists()]
                                                            
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

    read_tests(args.directory, args.latex, args.full)
