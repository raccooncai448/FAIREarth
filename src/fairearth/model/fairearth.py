import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from typing import Dict, Union, Optional, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
#print(os.getcwd())
#from model.utils import ExperimentConfig, GridDataset, ExperimentTracker, PrintLossCallback
from .utils.helpers import ExperimentConfig, GridDataset, ExperimentTracker, PrintLossCallback
from .evalUtils.evaluation import masked_losses

class FAIREarth:
    """Main research framework for training and evaluation"""
    def __init__(self, save_dir: str = "experiments"):
        self.tracker = ExperimentTracker(save_dir)
        self.models: Dict[str, pl.LightningModule] = {}
        self.datasets: Dict[str, Union[pl.LightningDataModule, np.ndarray]] = {}

    def add_model(self, name: str, model: pl.LightningModule):
        self.models[name] = model
        
    def add_dataset(self, name: str, dataset: Union[pl.LightningDataModule, np.ndarray], 
                    batch_size: int = 2048):
        """Add a dataset, either as a LightningDataModule or numpy array"""
        if isinstance(dataset, np.ndarray):
            #### FIX THIS
            # Create DataLoader for numpy array
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset = GridDataset(dataset[:train_size])
            val_dataset = GridDataset(dataset[train_size:])
            
            class GridDataModule(pl.LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.train_dataset = train_dataset
                    self.val_dataset = val_dataset
                    self.batch_size = batch_size

                def train_dataloader(self):
                    return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                   shuffle=True)

                def val_dataloader(self):
                    return DataLoader(self.val_dataset, batch_size=self.batch_size)

            dataset = GridDataModule()
        
        self.datasets[name] = dataset

    def train(self, model_name: str, dataset_name: str, 
              config: Optional[ExperimentConfig] = None, verbose: bool = True) -> Dict[str, Any]:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        if config is None:
            config = ExperimentConfig(model_name=model_name, dataset_name=dataset_name)

        model = self.models[model_name]
        datamodule = self.datasets[dataset_name]

        # Setup callbacks
        exp_dir = self.tracker.get_experiment_path(config)
        os.makedirs(exp_dir, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=exp_dir,
                filename=f"best",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stop_patience,
                mode="min"
            )
        ]
        if verbose:
            callbacks.append(PrintLossCallback())

        config.save(os.path.join(exp_dir, "config.json"))
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            callbacks=callbacks,
            default_root_dir=exp_dir
        )
        trainer.fit(model=model, datamodule=datamodule)

        metrics = {
            "best_val_loss": trainer.callback_metrics.get("val_loss", None).item(),
            "epochs_trained": trainer.current_epoch + 1,
            "model_name": model_name,
            "dataset_name": dataset_name
        }
        self.tracker.add_result(config, "metrics", metrics)
        self.tracker.update_result(config, metrics)

        return metrics

    def load_trained_model(self, experiment_name: str) -> pl.LightningModule:
        """Load a trained model from an experiment"""
        exp_dir = os.path.join(self.tracker.base_dir, experiment_name)
        config_path = os.path.join(exp_dir, "config.json")
        checkpoint_path = os.path.join(exp_dir, "best.ckpt")

        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            raise ValueError(f"Experiment {experiment_name} not found or incomplete")

        config = ExperimentConfig.load(config_path)
        model = self.models[config.model_name]
        model.load_from_checkpoint(checkpoint_path)
        
        return model

    def evaluate(self, model: pl.LightningModule, config: ExperimentConfig,
                    dataset_name: str = None, return_all: bool = False) -> Dict[str, float]:
        """Evaluate a model on a dataset using predictions"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        datamodule = self.datasets[dataset_name]
        
        # Create a trainer just for predictions
        trainer = pl.Trainer(accelerator='cpu', devices='auto')
        
        #
        #predictions = trainer.predict(model=model, datamodule=datamodule)
        predictions = [(torch.zeros((1800, 3600)), torch.zeros((1800, 3600)), torch.zeros((1800, 3600)))]

        
        all_predictions = np.concatenate([
            predict.cpu().numpy().flatten() 
            for predict, _, _ in predictions
        ])

        all_gt = np.concatenate([
            gt.cpu().numpy().flatten() 
            for _, _, gt in predictions
        ])

        if not return_all:
            subgroup_losses, analysis_results = masked_losses(all_predictions, all_gt)
            self.tracker.add_result(config, "subgroup_losses", subgroup_losses)
            #self.tracker.add_result(config, "analysis_results", analysis_results)
            
            return analysis_results
        
        return all_predictions, all_gt

    def list_experiments(self) -> pd.DataFrame:
        """List all experiments with their results"""
        data = []
        for exp_name, exp_data in self.tracker.results.items():
            row = {
                'experiment': exp_name,
                'model': exp_data['metrics']['model_name'],
                'dataset': exp_data['metrics']['dataset_name'], 
                'val_loss': exp_data['metrics']['best_val_loss'],
                'num_epochs': exp_data['metrics']['epochs_trained']
            }
            data.append(row)
        
        return pd.DataFrame(data)
