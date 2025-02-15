import torch
import os
import json
import uuid
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset
from lightning.pytorch.callbacks import Callback

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    model_name: str
    dataset_name: str
    timestamp: str = None
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 10
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @property
    def experiment_name(self) -> str:
        return f"{self.model_name}_{self.dataset_name}_{self.timestamp}"
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class GridDataset(Dataset):
    """Dataset for gridded data"""
    def __init__(self, data: np.ndarray, transform=None):
        self.data = torch.FloatTensor(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ExperimentTracker:
    """Tracks experiments and manages results"""
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.results_file = os.path.join(base_dir, f"results_{uuid.uuid4()}.json")
        os.makedirs(base_dir, exist_ok=True)
        self.load_results()

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = {}

    def save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def update_result(self, config: ExperimentConfig, metrics: dict):
        exp_name = config.experiment_name
        self.results[exp_name] = {
            'config': config.__dict__,
            'metrics': metrics
        }
        self.save_results()

    def add_result(self, config: ExperimentConfig, key: str, val):
        if not isinstance(val, dict):
            val = val.to_dict()
        exp_name = config.experiment_name
        if exp_name not in self.results:
            self.results[exp_name] = {}
        self.results[exp_name][key] = val
        self.save_results()

    def get_experiment_path(self, config: ExperimentConfig) -> str:
        return os.path.join(self.base_dir, config.experiment_name)
    
class PrintLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_losses.append(outputs['loss'].item())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_losses.append(outputs['val_loss'].item())

    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else None
        avg_val_loss = sum(self.val_losses) / len(self.val_losses) if self.val_losses else None
        
        if avg_val_loss and avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss

        print()
        print(f"Train Loss: {avg_train_loss:.4f}", end="")
        print(f", Val Loss: {avg_val_loss:.4f}", end="")
        print()

        self.train_losses = []
        self.val_losses = []
