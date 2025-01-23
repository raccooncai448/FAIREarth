import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
import pandas as pd
from data import (
    MetaDataModule, 
    ClimateDataModule,
    LandOceanDataModule,
    CheckerboardDataModule,
)
import yaml
from model.fairearth import FAIREarth
from locationencoder import LocationEncoder
import gc

with open('scripts/default_hparams.yaml', 'r') as file:
    default_hparams = yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description='Tune positional encoding parameters')
    parser.add_argument('--pe', type=str, choices=['wavelets', 'sphericalharmonics', 
                       'theory', 'spherecplus', 'spheremplus'],
                       help='Type of neural network to use')
    parser.add_argument('--dataset', type=str, default='landsea',
                       choices=['landsea', 'PRP', 'TASMAX', 'CO2', 
                                'checkerboard', 'landsea_original'],
                       help='Dataset to use for tuning')
    parser.add_argument('--cross_product', action='store_true', 
                        help='Run tuning across all PE methods and datasets')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to use for training')
    return parser.parse_args()

def get_hyperparameter(trial: optuna.trial.Trial, hparams):
    if hparams['architecture']['pe'] == 'wavelets':
        max_scale = trial.suggest_int("max_scale", 2, 5)
        max_possible_rotations = 400 // max_scale
        max_rotations = trial.suggest_int("max_rotations", 60, min(300, max_possible_rotations), step=10)

        hparams['embedding']["max_scale"] = max_scale
        hparams['embedding']["max_rotations"] = max_rotations
        hparams['embedding']['k_val'] = trial.suggest_float("k_val", 4, 10)
        hparams['embedding']['scale_factor'] = trial.suggest_float("scale_factor", 0.75, 1.25)
        hparams['embedding']['scale_shift'] = trial.suggest_float("scale_shift", 0, 1)
        hparams['embedding']['dilation_step'] = trial.suggest_float("dilation_step", 1, 8)
    elif hparams['architecture']['pe'] == 'sphericalharmonics':
        hparams['embedding']['legendre'] = trial.suggest_int("legendre", 10, 20, step=1)
    elif hparams['architecture']['pe'] in ["theory", "grid", "spherecplus", "spheremplus"]:
        hparams["min_radius"] = trial.suggest_int("min_radius", 1, 90, step=9)
        hparams["max_radius"] = 360
        hparams["frequency_num"] = trial.suggest_int("frequency_num", 16, 64, step=16)

    hparams['training']["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hparams['training']["wd"] = trial.suggest_float("wd", 1e-6, 1e-1, log=True)
    return hparams

def save_study_results(study, dataset_name, num_samples, positional_encoding_name):
    """Save study results to a single CSV file with unique identifiers"""
    results_dir = os.path.join("results", "hyperparameter_tuning")
    os.makedirs(results_dir, exist_ok=True)
    
    study_name = f"{dataset_name}-{positional_encoding_name}-{num_samples}"
    results_file = os.path.join(results_dir, f"{study_name}.csv")
    
    study.trials_dataframe().to_csv(results_file, index=False)

def tune(positional_encoding_name, neural_network_name, num_samples, model, 
         n_trials=40, dataset_name="landsea"):
    timeout = 24 * 60 * 60  # seconds

    hparams = default_hparams.copy()
    hparams['architecture']['pe'] = positional_encoding_name
    hparams['architecture']['nn'] = neural_network_name

    if dataset_name in ['PRP', 'TASMAX', 'CO2']:
        hparams['data']['meta_variable'] = dataset_name
        hparams['regression'] = True
    if dataset_name == 'checkerboard':
        hparams['dim_out'] = 4

    module = LocationEncoder(positional_encoding_name, neural_network_name, hparams)
    model.set_module(module)

    study_name = f"{dataset_name}-{positional_encoding_name}-{num_samples}"

    def objective(trial: optuna.trial.Trial, hparams, model) -> float:
        hparams = get_hyperparameter(trial, hparams)
        model.set_module(LocationEncoder(positional_encoding_name, neural_network_name, hparams))
        loss = model.fit(dataset_name, num_epochs=100, verbose=True)
        return loss

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )
    
    study.optimize(lambda x: objective(x, hparams, model), n_trials=n_trials, timeout=timeout)
    save_study_results(study, dataset_name, num_samples, positional_encoding_name)

    print(f"\nResults for {dataset_name} with {num_samples} samples:")
    print("Number of finished trials:", len(study.trials))
    print("\nBest trial:")
    trial = study.best_trial
    print("  Value:", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

def run_tuning(pe_method, dataset_name, fairearth, default_hparams, base_trials, num_samples):
    module = LocationEncoder(pe_method, 'siren', default_hparams)
    model = FAIREarth.initialize_model(input_size=None, preload_datasets=False, module=module)
    
    default_hparams['data']['num_samples'] = num_samples
    
    if dataset_name == 'landsea':
        default_hparams['data']['meta_variable'] = "LandSeaBinary"
        fairearth.add_dataset(MetaDataModule(default_hparams), 'landsea')
    elif dataset_name in ['PRP', 'TASMAX', 'CO2']:
        default_hparams['data']['meta_variable'] = dataset_name
        fairearth.add_dataset(ClimateDataModule(default_hparams), dataset_name)
    elif dataset_name == 'checkerboard':
        fairearth.add_dataset(CheckerboardDataModule(num_samples=num_samples, 
                                                   batch_size=default_hparams['training']['batch_size']),
                                                   dataset_name)
    elif dataset_name == 'landsea_original':
        fairearth.add_dataset(LandOceanDataModule(num_samples=num_samples, 
                                                batch_size=default_hparams['training']['batch_size']), 
                                                dataset_name)

    n_trials = base_trials * 4 if pe_method == 'wavelets' else base_trials
    tune(pe_method, 'siren', num_samples, model,
         n_trials=n_trials,
         dataset_name=dataset_name)
    gc.collect()

def main():
    args = parse_args()
    fairearth = FAIREarth()
    base_trials = 30
    
    if args.cross_product:
        pe_methods = ['wavelets', 'sphericalharmonics', 'theory', 'spherecplus', 'spheremplus']
        datasets = ['landsea', 'PRP', 'TASMAX', 'CO2']
        
        for pe_method in pe_methods:
            for dataset_name in datasets:
                print(f"Running tuning for PE: {pe_method}, Dataset: {dataset_name}")
                run_tuning(pe_method, dataset_name, fairearth, default_hparams.copy(), 
                          base_trials, args.num_samples)
    else:
        run_tuning(args.pe, args.dataset, fairearth, default_hparams.copy(), 
                  base_trials, args.num_samples)

if __name__ == '__main__':
    main()