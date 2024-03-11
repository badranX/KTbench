from datetime import datetime
import yamld
from pathlib import Path
import torch
import json
from pathlib import Path
import re
import time


KTBENCH_FOLDER = ".ktbench"

class LogsHandler:
    def __init__(self, config, checkpoint_parent_folder=None):
        current_directory = Path.cwd()
        if checkpoint_parent_folder:
            self.checkpoint_parent_folder = checkpoint_parent_folder
        else:
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
            tmp= {k: v for k,v in vdir(obj).items() if isinstance(v, Number) or
                     isinstance(v, bool) or
                     isinstance(v, str) or
                     isinstance(v, list) or
                     isinstance(v, tuple) 
                     }
            tmp.update({k:list(v) for k,v in tmp.items() if isinstance(v, tuple)})
            return tmp

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


def read_tests(directory_path, full=False):
    def extract_timestamp(folder_name):
        return time.mktime(time.strptime(folder_name, '%Ss%Mm%Hh-%dD%mM%YY'))

    timestamp_pattern = re.compile(r'\d{2}s\d{2}m\d{2}h-\d{2}D\d{2}M\d{4}Y')

    directory_path = Path(directory_path)

    timestamp_folders = {}
    meta_data = {}

    for dsdir in directory_path.iterdir():
        if dsdir.is_dir():
            #dataset dir
            for modeldir in dsdir.iterdir():
                tmp = sorted([timedir for timedir in modeldir.iterdir() if timedir.is_dir()],
                                                          key=lambda x: extract_timestamp(x.name),
                                                          reverse=True)

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
        for k, timel in meta_data.items():
            print('### ', k)
            for traintime, df in timel:
                print('##### ', traintime)
                print(df.mean())


    return timestamp_folders


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Simple Args Parser")

    parser.add_argument("directory", nargs="?", default=Path.cwd(), type=Path,
                        help="The directory to process (default: current working directory)")

    parser.add_argument("--full", action="store_true", help="Include full processing")

    args =  parser.parse_args()

    read_tests(args.directory, args.full)
