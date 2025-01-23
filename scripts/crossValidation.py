import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
import uuid
import json
from datetime import datetime
from data import (
    MetaDataModule, 
    ClimateDataModule
)
import yaml
from model.fairearth import FAIREarth
from locationencoder import LocationEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model with given hyperparameters')
    parser.add_argument('--pe', type=str, required=True,
                       choices=['wavelets', 'sphericalharmonics', 'theory', 'spherecplus', 'spheremplus'],
                       help='Type of positional encoding')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['landsea', 'PRP', 'TASMAX', 'CO2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--num_samples', type=int, required=True,
                       help='Number of samples to use for training')
    
    # Wavelets-specific params
    parser.add_argument('--max_scale', type=int)
    parser.add_argument('--max_rotations', type=int)
    parser.add_argument('--k_val', type=float)
    #parser.add_argument('--scale_factor', type=float)
    #parser.add_argument('--scale_shift', type=float)
    #parser.add_argument('--dilation_step', type=float)
    
    # Spherical harmonics params
    parser.add_argument('--legendre', type=int)
    
    # Theory/Grid/Sphere params
    parser.add_argument('--min_radius', type=int)
    parser.add_argument('--frequency_num', type=int)
    
    # Training params
    parser.add_argument('--lr', type=float, required=True,
                       help='Learning rate')
    parser.add_argument('--wd', type=float, required=True,
                       help='Weight decay')
    
    return parser.parse_args()

def setup_hparams(args):
    # Load default hyperparameters
    with open('scripts/default_hparams.yaml', 'r') as file:
        hparams = yaml.safe_load(file)
    
    # Update with command line arguments
    hparams['architecture']['pe'] = args.pe
    hparams['architecture']['nn'] = 'siren'  # Using SIREN as default neural network
    hparams['data']['num_samples'] = args.num_samples
    hparams['training']['lr'] = args.lr
    hparams['training']['wd'] = args.wd
    
    # Set dataset-specific parameters
    if args.dataset in ['PRP', 'TASMAX', 'CO2']:
        hparams['data']['meta_variable'] = args.dataset
        hparams['regression'] = True
    
    # Set PE-specific parameters
    if args.pe == 'wavelets':
        hparams['embedding'].update({
            'max_scale': args.max_scale,
            'max_rotations': args.max_rotations,
            'k_val': args.k_val,
            #'scale_factor': args.scale_factor,
            #'scale_shift': args.scale_shift,
            #'dilation_step': args.dilation_step
        })
    elif args.pe == 'sphericalharmonics':
        hparams['embedding']['legendre'] = args.legendre
    elif args.pe in ['theory', 'grid', 'spherecplus', 'spheremplus']:
        hparams.update({
            'min_radius': args.min_radius,
            'max_radius': 360,
            'frequency_num': args.frequency_num
        })
    
    return hparams

def evaluate_model(args, hparams):
    configs_dir = os.path.join('results', 'cross_validation', 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    
    unique_id = str(uuid.uuid4())[:8]
    config_filename = f"{args.dataset}_{args.pe}_{args.num_samples}_{unique_id}.json"
    config_filepath = os.path.join(configs_dir, config_filename)
    
    with open(config_filepath, 'w') as f:
        json.dump(hparams, f, indent=2)
    
    print(f"Config saved to: {config_filepath}")
    
    fairearth = FAIREarth()
    module = LocationEncoder(args.pe, 'siren', hparams)
    model = FAIREarth.initialize_model(input_size=None, preload_datasets=False, module=module)
    dataset_name = args.dataset
    if dataset_name== 'landsea':
        hparams['data']['meta_variable'] = "LandSeaBinary"
        fairearth.add_dataset(MetaDataModule(hparams), 'landsea')
    elif dataset_name in ['PRP', 'TASMAX', 'CO2']:
        hparams['data']['meta_variable'] = dataset_name
        fairearth.add_dataset(ClimateDataModule(hparams), dataset_name)
    
    results_dir = os.path.join('results', 'cross_validation')
    results_filename = f"{unique_id}"
    results_filepath = os.path.join(results_dir, results_filename)
    model.fit(args.dataset, num_epochs=300, verbose=True)
    model.evaluate(args.dataset, save_file=results_filepath)
    

def main():
    args = parse_args()
    hparams = setup_hparams(args)
    evaluate_model(args, hparams)

if __name__ == '__main__':
    main()