import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
import pandas as pd

'''
TODO:
- Add index into the 'values' returned
'''

DATA_DIR = "datasets/test_datasets"
climate_data = pd.read_csv(DATA_DIR + '/new_climate.csv')
climate_data_tas = climate_data.filter(regex=f'^{"TAS"}_')
climate_data_tasmax = climate_data.filter(regex=f'^{"TASMAX"}_')
climate_data_prp = climate_data.filter(regex=f'^{"PR"}_')
climate_data_CO2 = climate_data['CO2']

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def find_closest_values(lons, lats, grid, time=-1):
    lons = np.array(lons)
    lats = np.array(lats)

    lons = np.where(lons > 180, lons - 360, lons)
    lons = np.where(lons < -180, lons + 360, lons)
    
    lat_grid = np.linspace(-89.95, 89.95, 1800)
    lon_grid = np.linspace(-179.95, 179.95, 3600)

    lat_indices = np.clip(((89.95 - lats) / 0.1).astype(int), 0, 1800)
    lon_indices = np.clip(((lons + 179.95) / 0.1).astype(int), 0, 3600)

    if time == -1:
        closest_values = grid[:, lat_indices, lon_indices].flatten()
    else:
        if grid.ndim == 2:
            closest_values = grid[lat_indices, lon_indices].flatten()
        else:
            closest_values = grid[time, lat_indices, lon_indices].flatten()
    
    return closest_values

def get_data_points(N=5000, seed=0, grid=False, meta_variable='TAS', time=-1):
    variable_df = None
    if meta_variable == 'TAS':
        variable_df = climate_data_tas
    elif meta_variable == 'TASMAX':
        variable_df = climate_data_tasmax
    elif meta_variable == 'PRP':
        variable_df = climate_data_prp
    elif meta_variable == 'CO2':
        variable_df = climate_data_CO2
    else:
        print("Wrong meta variable outlined, reverting back to PRP")
        variable_df = climate_data_prp
    
    if meta_variable == 'CO2':
        gridData = variable_df.values.reshape(1800, 3600)
    else:
        gridData = variable_df.values.T.reshape(12, 1800, 3600)
    # Do normalization
    gridData = (gridData - np.min(gridData)) / (np.max(gridData) - np.min(gridData)) * 10
    # bins = np.linspace(0, 1, 11)
    # binned_data = np.digitize(gridData, bins) - 1
    # gridData = binned_data
    if meta_variable == 'CO2':
        gridData = np.flip(gridData)
        gridData = np.hstack((gridData[:,1800:], gridData[:,:1800]))

    time_index = None
    if grid:
        lon_coords, lat_coords = np.linspace(-179.95, 179.95, 3600), np.linspace(89.95, -89.95, 1800)
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        coord_grid = np.stack((lat_grid, lon_grid), axis=-1)

        lons, lats = coord_grid[:,:,1].flatten(), coord_grid[:,:,0].flatten()
        if time == -1:
            time_index = np.repeat(np.arange(1, 13), len(lons))
            lons, lats = np.tile(lons, 12), np.tile(lats, 12)
            values = gridData.reshape(12, -1).flatten()
        else:
            if gridData.ndim == 2:
                values = gridData.flatten()
            else:
                values = gridData[time].flatten()
    else:
        rng = np.random.RandomState(seed)
        x, y, z = rng.normal(size=(3, N))
        az, el, _ = cart2sph(x, y, z)
        lons, lats = np.rad2deg(az), np.rad2deg(el)
        values = find_closest_values(lons, lats, gridData, time=time)
        if time == -1:
            time_index = np.repeat(np.arange(1, 13), len(lons))
            lons, lats = np.tile(lons, 12), np.tile(lats, 12)
    return (lons, lats, time_index), values

def get_data(N=5000, seed=0, grid=False, meta_variable='TAS', time=-1):
    #import pdb
    #pdb.set_trace()
    (lons, lats, time_index), values = get_data_points(N, seed, grid=grid, meta_variable=meta_variable, time=time)
    if time == -1:
        lonlats = torch.stack([torch.tensor(lons), torch.tensor(lats), torch.tensor(time_index)], dim=1)
    else:
        lonlats = torch.stack([torch.tensor(lons), torch.tensor(lats)], dim=1)
    #import pdb
    #pdb.set_trace()
    #values = torch.from_numpy(values).unsqueeze(-1).to(torch.float64) / 10
    values = torch.from_numpy(values).unsqueeze(-1).to(torch.float64)

    return lonlats, values

# Combined Climate Data Module
class ClimateDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.num_samples = hparams['data']['num_samples']
        self.batch_size=hparams['training']['batch_size']
        self.meta_variable=hparams['data']['meta_variable']
        self.time = hparams['data']['time_idx']
        self.train_seed, self.valid_seed = hparams['data']['train_seed'], hparams['data']['valid_seed']
        self.pin_memory = False

    def setup(self, stage: str):
        self.train_ds = TensorDataset(*get_data(self.num_samples, seed=self.train_seed, meta_variable=self.meta_variable, time=self.time))
        self.valid_ds = TensorDataset(*get_data(int(self.num_samples / 4), seed=self.valid_seed, meta_variable=self.meta_variable, time=self.time))
        self.evalu_ds = TensorDataset(*get_data(self.num_samples, seed=self.valid_seed, grid=True, meta_variable=self.meta_variable, time=self.time))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)
    
    def predict_dataloader(self):
        return DataLoader(self.evalu_ds, batch_size=10000, shuffle=False, pin_memory=self.pin_memory)



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
    import pdb
    pdb.set_trace()

    use_grid = True
    time = -1
    (lons, lats, time_index), values = get_data_points(N=500, seed=2, grid=use_grid, 
                                                       time=time, meta_variable='CO2')
    if time == -1:
        lons, lats, values = lons[:(1800*3600)], lats[:(1800*3600)], values[:(1800*3600)]
    if use_grid:
        lons, lats, values = lons[::260], lats[::260], values[::260]
    plot_coordinates(lons, lats, values)
    print(np.min(lons), np.max(lons), np.min(lats), np.max(lats))
    print(values)
    print(lons.shape)
        
