import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from model.fairearth import FAIREarth
import numpy as np
import torch.nn as nn
import pandas as pd
from data import (
    MetaDataModule,
    LandOceanDataModule
)
import torch
from locationencoder import LocationEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

bce_loss = lambda pred, gt: -np.mean(gt * np.log(pred + 1e-7) + (1 - gt) * np.log(1 - pred + 1e-7))

# Initializing base model
with open('scripts/default_hparams.yaml', 'r') as file:
    hparams = yaml.safe_load(file)

fairearth = FAIREarth()
# To initialize model, first create a Lightning module
# Preloaded datasets are available for inference if needed
# Using the same model, we can set a different module (flushes out previous module data)
module = LocationEncoder('wavelets', 'siren', hparams)
model = FAIREarth.initialize_model(input_size=None, preload_datasets=False, module=module)
fairearth.add_dataset(LandOceanDataModule(num_samples=100, 
                                        batch_size=hparams['training']['batch_size']), 
                                        'landsea_original')

model.fit('landsea_original', 200, verbose=False)
# Saves per-group evaluation results
predictions, gt = model.evaluate('landsea_original', save_file=None, verbose=False, return_all=True)
#import pdb
#pdb.set_trace()
pred_sigmoid = sigmoid(predictions)
losses = bce_loss(pred_sigmoid, gt.astype(float))
        
# Save to CSV
df = pd.DataFrame({
    'prediction': pred_sigmoid,
    'ground_truth': gt,
    'bce_loss': losses
})
df.to_csv(f'bce_losses_SH.csv', index=False)


model.set_module(LocationEncoder('sphericalharmonics', 'siren', hparams))
model.fit('landsea_original', 200, verbose=False)
predictions, gt = model.evaluate('landsea_original', save_file=None, verbose=False, return_all=True)

pred_sigmoid = sigmoid(predictions)
losses = bce_loss(pred_sigmoid, gt.astype(float))
        
# Save to CSV
df = pd.DataFrame({
    'prediction': pred_sigmoid,
    'ground_truth': gt,
    'bce_loss': losses
})
df.to_csv(f'bce_losses_SW.csv', index=False)

# TODO: incorporate visualization scripts (i.e. heatmap, error plots) as methods

