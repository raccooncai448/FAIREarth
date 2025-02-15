import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append('.')
#sys.path.append('src')

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append('src')

import json, yaml
from datetime import datetime
from fairearth.model.fairearth import FAIREarth
#from model.fairearth import FAIREarth
import numpy as np
import torch.nn as nn
import lightning as pl
from data import (
    MetaDataModule,
    LandOceanDataModule,
    ClimateDataModule
)
from fairearth.model.utils import ExperimentConfig
import torch
from fairearth.locationencoder import LocationEncoder

# Initialize framework
framework = FAIREarth()

# Add a custom model
with open('scripts/default_hparams.yaml', 'r') as file:
    hparams = yaml.safe_load(file)
module = LocationEncoder('wavelets', 'siren', hparams)
framework.add_model("my_model", module)
framework.add_dataset('my_dataset', MetaDataModule(hparams))

# Train
config = ExperimentConfig(
    model_name="my_model",
    dataset_name="my_dataset",
    max_epochs=10
)
metrics = framework.train("my_model", "my_dataset", config)

# Load and evaluate
trained_model = framework.load_trained_model(config.experiment_name)
results = framework.evaluate(
    trained_model,
    config,
    dataset_name="my_dataset"
)

print("\nEvaluation Results:")
print(results)

# View all experiments
experiments_df = framework.list_experiments()
print("\nAll Experiments:")
print(experiments_df)
