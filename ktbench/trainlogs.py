from datetime import datetime
from pathlib import Path
import torch
import json
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
        
    def train_starts(self, model_name):
        self.timestamp = datetime.now().strftime("%Ss%Mm%Hh-%dD%mM%YY")
        append = getattr(self.cfg, 'append2logdir', '')
        self.current_checkpoint_folder = self.dataset_window_folder/(model_name + append)/self.timestamp

        self.current_checkpoint_folder.mkdir(parents=True, exist_ok=True)

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

    def load_best_model(self, ModelClass, kfold):
        best_model_filename = self.current_checkpoint_folder/f"best_model_fold_{kfold}.pth"

        # Load best model state
        checkpoint = torch.load(best_model_filename)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def save_best_model(self, model, best_epoch, kfold):
        best_model_filename = self.current_checkpoint_folder/f"best_model_fold_{kfold}.pth"

        # Save best model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': best_epoch,
            'config': model.cfg,
            'prm': model.prm
        }, best_model_filename)
