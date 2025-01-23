import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import pandas as pd
from matplotlib.colors import Normalize

DATA_DIR = "datasets/test_datasets"
metadata = pd.read_csv(DATA_DIR + '/metadata.csv', low_memory=False)

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def find_closest_values(lons, lats, grid):
    lons = np.array(lons)
    lats = np.array(lats)

    lons = np.where(lons > 180, lons - 360, lons)
    lons = np.where(lons < -180, lons + 360, lons)
    
    lat_grid = np.linspace(-89.95, 89.95, 1800)
    lon_grid = np.linspace(-179.95, 179.95, 3600)

    lat_indices = np.clip(((89.95 - lats) / 0.1).astype(int), 0, 1800)
    lon_indices = np.clip(((lons + 179.95) / 0.1).astype(int), 0, 3600)

    closest_values = grid[lat_indices, lon_indices]
    
    return closest_values

def get_data_points(N=5000, seed=0, sphericaluniform=True, grid=False, meta_variable='LandSeaBinary', save=False):
    gridData = np.array(metadata[meta_variable].values).reshape(1800, 3600)
    gridData = np.flip(gridData, axis=0)
    gridData = np.concatenate([gridData[:,1800:], gridData[:,:1800]], axis=1)
    gridData = (gridData - np.min(gridData)) / (np.max(gridData) - np.min(gridData))

    if grid:
        lon_coords, lat_coords = np.linspace(-179.95, 179.95, 3600), np.linspace(89.95, -89.95, 1800)
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        coord_grid = np.stack((lat_grid, lon_grid), axis=-1)

        lons, lats = coord_grid[:,:,1].flatten(), coord_grid[:,:,0].flatten()
        values = gridData.flatten()
    else:
        rng = np.random.RandomState(seed)
        x, y, z = rng.normal(size=(3, N))
        az, el, _ = cart2sph(x, y, z)
        lons, lats = np.rad2deg(az), np.rad2deg(el)
        if save:
            np.save(f'longitudes_{N}.npy', lons)
            np.save(f'latitudes_{N}.npy', lats)
        values = find_closest_values(lons, lats, gridData)

    return (lons, lats), values

def get_data(N=5000, seed=0, grid=False, meta_variable='LandSeaBinary', save=False):
    (lons, lats), values = get_data_points(N, seed, grid=grid, meta_variable=meta_variable, save=save)
    lonlats = torch.stack([torch.tensor(lons), torch.tensor(lats)], dim=1).to(torch.float32)
    values = torch.from_numpy(values).unsqueeze(-1).to(torch.float64)

    return lonlats, values

class MetaDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.num_samples = hparams['data']['num_samples']
        self.batch_size=hparams['training']['batch_size']
        self.meta_variable=hparams['data']['meta_variable']
        self.train_seed, self.valid_seed = hparams['data']['train_seed'], hparams['data']['valid_seed']
        self.pin_memory = True
        self.num_workers = 0

    def setup(self, stage: str):
        self.train_ds = TensorDataset(*get_data(self.num_samples, seed=self.train_seed, meta_variable=self.meta_variable, save=False))
        self.valid_ds = TensorDataset(*get_data(int(self.num_samples / 4), seed=self.valid_seed, meta_variable=self.meta_variable))
        self.evalu_ds = TensorDataset(*get_data(self.num_samples, grid=True, meta_variable=self.meta_variable))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
                          shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=40000, 
                          shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
    

'''
TESTING PURPOSES
'''

def plot_coordinates(lons, lats, values):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(12, 8))
    world.plot(ax=ax, color='lightgrey', edgecolor='black')

    scatter = ax.scatter(lons, lats, c=values, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Values', orientation='vertical', pad=0.05)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Value Distribution on World Map')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    use_grid = False
    (lons, lats), values = get_data_points(N=500, seed=2, grid=use_grid)
    if use_grid:
        lons, lats, values = lons[::2024], lats[::2024], values[::2024]
    plot_coordinates(lons, lats, values)
    print(np.min(lons), np.max(lons), np.min(lats), np.max(lats))
    print(values)
    print(lons.shape)
