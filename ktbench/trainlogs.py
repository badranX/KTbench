from datetime import datetime
from pathlib import Path
import torch
import json

class LogsHandler:
    def __init__(self, config, checkpoint_main_folder=None):

        self.checkpoint_main_folder  = Path.cwd() if not checkpoint_main_folder else Path(checkpoint_main_folder)
        self.datasetname = config.dataset_name
        self.windowsize = config.window_size
        self.checkpoint_folder = self.checkpoint_main_folder / f"{self.datasetname}_{self.windowsize}"
        self.timestamp = datetime.now().strftime("%S-%M-%H--%d-%m-%Y")
        self.current_checkpoint_folder = self.checkpoint_folder / self.timestamp

        # Create necessary directories
        #self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
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
        if len(list(self.checkpoint_folder.glob("*"))) > 1:
            previous_checkpoint_folder = sorted(self.checkpoint_folder.glob("*"))[0]
            previous_checkpoint_folder.rmdir()

    def load_checkpoint(self, ModelClass, optimizer):
        latest_checkpoint_folder = sorted(self.checkpoint_folder.glob("*"))[-1]
        latest_checkpoint_path = latest_checkpoint_folder / "checkpoint.pth"

        # Load model and optimizer state
        checkpoint = torch.load(latest_checkpoint_path)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, checkpoint['epoch']

    def load_best_model(self, ModelClass):
        best_model_filename = self.checkpoint_folder / "best_model.pth"

        # Load best model state
        checkpoint = torch.load(best_model_filename)
        model = ModelClass(cfg=checkpoint['config'], params=checkpoint['prm'])
        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def save_best_model(self, model, best_epoch):
        best_model_filename = self.checkpoint_folder / "best_model.pth"

        # Save best model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': best_epoch,
            'config': model.cfg,
            'prm': model.prm
        }, best_model_filename)
