import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score, jaccard_score, mean_squared_error, log_loss
from typing import Tuple, Dict, List
import pickle
import os

# Global metadata
from importlib import resources
import pandas as pd

def load_metadata():
    with resources.files('fairearth').joinpath('datasets/test_datasets/metadata.csv').open('r') as f:
        return pd.read_csv(f, index_col=0, low_memory=False)
        
#transformed_metadata = pd.read_csv('datasets/test_datasets/metadata.csv', index_col=0, low_memory=False)
transformed_metadata = load_metadata()

# Transform spatial data globally
land_sea_binary = np.array(transformed_metadata['LandSeaBinary'].values).reshape(1800, 3600)
land_sea_binary = np.flip(land_sea_binary, axis=0)
land_sea_binary = np.concatenate([land_sea_binary[:,1800:], land_sea_binary[:,:1800]], axis=1)
land_sea_binary = land_sea_binary.flatten()

size_data = np.array(transformed_metadata['SizeData'].values).reshape(1800, 3600)
size_data = np.flip(size_data, axis=0)
size_data = np.concatenate([size_data[:,1800:], size_data[:,:1800]], axis=1)
size_data = size_data.flatten()

coastline_binned = np.array(transformed_metadata['CoastlineBinned'].values).reshape(1800, 3600)
coastline_binned = np.flip(coastline_binned, axis=0)
coastline_binned = np.concatenate([coastline_binned[:,1800:], coastline_binned[:,:1800]], axis=1)
coastline_binned = coastline_binned.flatten()

country_data = np.array(transformed_metadata['Country'].values).reshape(1800, 3600)
country_data = np.flip(country_data, axis=0)
country_data = np.concatenate([country_data[:,1800:], country_data[:,:1800]], axis=1)
country_data = country_data.flatten()

# Create transformed metadata DataFrame
transformed_metadata = pd.DataFrame({
    'LandSeaBinary': land_sea_binary,
    'SizeData': size_data,
    'CoastlineBinned': coastline_binned,
    'Country': country_data
})

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def sigmoid(x):
    """Apply sigmoid function to input."""
    return 1 / (1 + np.exp(-x))

def create_base_mask_tensor() -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Create base masks (non-country masks) and return tensor, mapping and names.
    """
    masks = []
    mask_names = []
    
    # Land/Sea masks
    masks.extend([
        transformed_metadata['LandSeaBinary'] == 0,                                    # Land
        transformed_metadata['LandSeaBinary'] == 1,                                    # Sea
        (transformed_metadata['LandSeaBinary'] == 0) & 
        (transformed_metadata['SizeData'] <= 100),                                     # Island
        (transformed_metadata['LandSeaBinary'] == 0) & 
        (transformed_metadata['SizeData'] > 100),                                      # Mainland
    ])
    mask_names.extend(['Land', 'Sea', 'Island', 'Mainland'])
    
    # Coast masks
    coast_bins = transformed_metadata['CoastlineBinned'].dropna().unique()
    for bin_value in coast_bins:
        masks.append(transformed_metadata['CoastlineBinned'] == bin_value)
        mask_names.append(f'Coast_Bin_{int(bin_value)}')
    
    mask_tensor = np.stack(masks, axis=0)
    mask_mapping = {name: idx for idx, name in enumerate(mask_names)}
    
    return mask_tensor, mask_mapping, mask_names

def get_country_batches(min_points: int = 500, batch_size: int = 5) -> List[List[str]]:
    """
    Get list of country batches for those with more than min_points.
    """
    country_counts = transformed_metadata['Country'].value_counts()
    eligible_countries = country_counts[country_counts >= min_points].index.tolist()
    
    print(f"\nCountry Selection Summary:")
    print(f"Countries with ≥{min_points} points: {len(eligible_countries)}")
    print(f"Will process in batches of {batch_size}")
    
    return [eligible_countries[i:i + batch_size] 
            for i in range(0, len(eligible_countries), batch_size)]

def process_country_batch(countries: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Create masks for a batch of countries.
    """
    masks = []
    mask_names = []
    
    for country in countries:
        mask = transformed_metadata['Country'] == country
        if mask.sum() > 0:  # Only add if mask has points
            masks.append(mask)
            mask_names.append(f'Country_{country}')
            
    return masks, mask_names

def vectorized_binary_metrics(pred_probs: np.ndarray, gt: np.ndarray, 
                            mask_tensor: np.ndarray, mask_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute binary metrics for all masks simultaneously.
    """
    metrics = {'CrossEntropy': {}, 'Accuracy': {}, 'IoU': {}}
    
    for idx, (mask, name) in enumerate(zip(mask_tensor, mask_names)):
        if mask.sum() == 0:
            continue
            
        try:
            metrics['CrossEntropy'][name] = log_loss(
                gt[mask], pred_probs[mask], labels=[0, 1]
            )
            metrics['Accuracy'][name] = accuracy_score(
                gt[mask], pred_probs[mask] > 0.5
            )
            metrics['IoU'][name] = jaccard_score(
                gt[mask], pred_probs[mask] > 0.5, average='binary', labels=[0, 1]
            )
        except ValueError:
            # Handle edge cases
            pred_class = (pred_probs[mask] > 0.5).astype(float)
            if np.all(gt[mask] == 0):
                metrics['IoU'][name] = float(np.all(pred_class == 0))
                metrics['CrossEntropy'][name] = 0.0 if np.all(pred_class == 0) else float('inf')
            else:
                metrics['IoU'][name] = float(np.all(pred_class == 1))
                metrics['CrossEntropy'][name] = 0.0 if np.all(pred_class == 1) else float('inf')
            metrics['Accuracy'][name] = float(np.all(pred_class == gt[mask]))
            
    return metrics

def vectorized_regression_metrics(predictions: np.ndarray, gt: np.ndarray, 
                                mask_tensor: np.ndarray, mask_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute regression metrics for all masks simultaneously.
    """
    metrics = {'MSE': {}}
    
    for idx, (mask, name) in enumerate(zip(mask_tensor, mask_names)):
        if mask.sum() == 0:
            continue
        metrics['MSE'][name] = mean_squared_error(gt[mask], predictions[mask])
            
    return metrics

def masked_losses(predictions: np.ndarray, gt: np.ndarray,
                 cache_path: str = 'datasets/mask_cache.pkl') -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate losses for different subgroups using batched operations for countries.
    """
    is_binary = len(np.unique(gt)) == 2
    if is_binary:
        pred_probs = sigmoid(predictions)
    
    # Get base masks (non-country masks)
    base_tensor, base_mapping, base_names = create_base_mask_tensor()
    
    # Calculate base metrics
    if is_binary:
        metrics_dict = vectorized_binary_metrics(pred_probs, gt, base_tensor, base_names)
        overall_metrics = {
            'CrossEntropy': log_loss(gt, pred_probs, labels=[0, 1]),
            'Accuracy': accuracy_score(gt, pred_probs > 0.5),
            'IoU': jaccard_score(gt, pred_probs > 0.5, average='binary', labels=[0, 1])
        }
    else:
        metrics_dict = vectorized_regression_metrics(predictions, gt, base_tensor, base_names)
        overall_metrics = {'MSE': mean_squared_error(gt, predictions)}
    
    # Process country batches
    country_batches = get_country_batches()
    for batch_idx, country_batch in enumerate(country_batches):
        print(f"Processing country batch {batch_idx + 1}/{len(country_batches)}")
        
        # Create masks for this batch
        country_masks, country_names = process_country_batch(country_batch)
        
        if country_masks:  # If we have valid masks
            country_tensor = np.stack(country_masks, axis=0)
            
            # Calculate metrics for this batch
            if is_binary:
                batch_metrics = vectorized_binary_metrics(pred_probs, gt, country_tensor, country_names)
            else:
                batch_metrics = vectorized_regression_metrics(predictions, gt, country_tensor, country_names)
            
            # Update metrics dictionary
            for metric_name in batch_metrics:
                metrics_dict[metric_name].update(batch_metrics[metric_name])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(metrics_dict)
    
    # Analysis calculations
    metadata_analysis = {}
    for metric in metrics_dict.keys():
        metric_series = results_df[metric]
        
        if metric in ['CrossEntropy', 'MSE']:
            metadata_analysis[f'{metric}_analysis'] = {
                'best_group': metric_series.idxmin(),
                'worst_group': metric_series.idxmax(),
                'best_value': metric_series.min(),
                'worst_value': metric_series.max(),
                'std_dev': metric_series.std(),
                'median': metric_series.median()
            }
        else:
            metadata_analysis[f'{metric}_analysis'] = {
                'best_group': metric_series.idxmax(),
                'worst_group': metric_series.idxmin(),
                'best_value': metric_series.max(),
                'worst_value': metric_series.min(),
                'std_dev': metric_series.std(),
                'median': metric_series.median()
            }
        
        gaps = {
            'land_sea_gap': abs(metrics_dict[metric].get('Land', 0) - metrics_dict[metric].get('Sea', 0)),
            'island_mainland_gap': abs(metrics_dict[metric].get('Island', 0) - metrics_dict[metric].get('Mainland', 0)),
        }
        metadata_analysis[f'{metric}_gaps'] = gaps
    
    # Collect subgroup counts using transformed metadata
    subgroup_counts = {}
    # Base counts
    for name, mask in zip(base_names, base_tensor):
        subgroup_counts[name] = mask.sum()
    # Country counts
    for metric in metrics_dict.values():
        for name in metric.keys():
            if name.startswith('Country_'):
                country = name.split('_')[1]
                mask = transformed_metadata['Country'] == country
                subgroup_counts[name] = mask.sum()
    
    analysis_results = {
        'overall_metrics': overall_metrics,
        'metadata_analysis': metadata_analysis,
        'subgroup_count': subgroup_counts
    }
    
    return results_df, convert_to_serializable(analysis_results)
