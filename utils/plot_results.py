import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.basemap import Basemap
from pyproj import CRS
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

def is_island(lat, lon, world, max_area_km2=10000):
    point = Point(lon, lat)
    intersecting = world[world.intersects(point)]
    
    if intersecting.empty:
        return False  # Point is in the ocean
    
    area_km2 = intersecting.to_crs(CRS.from_epsg(6933)).area.sum() / 1e6
    return area_km2 <= max_area_km2

def create_island_map(title=None, bds=[-180, -90, 180, 90], plot_points=None):
    # Load world map data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Create the figure and Basemap
    fig = plt.figure(figsize=(20, 10))
    map = Basemap(*bds)
    ax = plt.gca()

    # Create a grid for coloring
    lons = np.linspace(bds[0], bds[2], 360)
    lats = np.linspace(bds[1], bds[3], 180)
    lons, lats = np.meshgrid(lons, lats)

    # Initialize the color grid (0: ocean, 1: island, 0.5: non-island landmass)
    color_grid = np.zeros(lons.shape)

    # Vectorize the is_land function
    vect_is_land = np.vectorize(map.is_land)

    # Check if each point is land
    land_mask = vect_is_land(lons, lats)

    # For each land point, check if it's an island
    for i in range(lons.shape[0]):
        for j in range(lons.shape[1]):
            if land_mask[i, j]:
                is_island_result = is_island(lats[i, j], lons[i, j], world)
                color_grid[i, j] = 1 if is_island_result else 0.5

    # Plot the colored grid
    map.imshow(color_grid, origin="lower", interpolation="none", cmap="RdBu_r", vmin=0, vmax=1)

    # Plot points if provided
    if plot_points is not None:
        x, y = map(plot_points[:, 0], plot_points[:, 1])
        map.scatter(x, y, c="red", zorder=5)

    # Draw coastlines
    map.drawcoastlines(linewidth=0.5)

    # Set labels and title
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if title is not None:
        ax.set_title(title)

    return fig


def plot_predictions(spatialencoder,
                     predictions, 
                     bds=[-180,-90,180,90], 
                     #degrees_per_pixel = 1,
                     title=None, show=True, savepath=None, plot_points=None, class_idx=None,
                     save_globe=True,
                     save_islands=True):
    """
    convenience function to plot a regular grid of lon lat coordinates
    """
    device = spatialencoder.device

    ideal_degrees_per_pixel_lat = 180 / (bds[3] - bds[1]) 
    ideal_degrees_per_pixel_lon = 360 / (bds[2] - bds[0]) 
    
    degrees_per_pixel = max(ideal_degrees_per_pixel_lat,ideal_degrees_per_pixel_lat)
    num_pix_lon = int((bds[2] - bds[0]) * degrees_per_pixel)
    num_pix_lat = int((bds[3] - bds[1]) * degrees_per_pixel)
    lon = torch.tensor(np.linspace(bds[0], bds[2], num_pix_lon ), device=device)
    lat = torch.tensor(np.linspace(bds[1], bds[3], num_pix_lat), device=device)
    lons, lats = torch.meshgrid(lon, lat)

    # ij indexing to xy indexing
    lons, lats = lons.T, lats.T

    lonlats = torch.stack([lons, lats], dim=-1).view(-1, 2)

    spatialencoder.eval()
    if spatialencoder.regression:
        print("Using regression")
        #Y = spatialencoder(lonlats)
        Y = torch.tensor(predictions)
    else:
        #Y = torch.sigmoid(spatialencoder(lonlats))
        Y = torch.sigmoid(torch.tensor(predictions))

    if class_idx is not None:
        Y = Y[:, class_idx].unsqueeze(-1)

    Y = Y.view(1800, 3600)
    #Y = Y.view(num_pix_lat, num_pix_lon)

    # if not binary show predictions instead of probabilities
    if not Y.size(-1) == 1:
        y = Y.argmax(-1)
    else:
        print("Binary")
        y = Y

    # if savepath is not None and save_globe:
    #     fig = draw_globe(y, lonlats, plot_points, title)
    #     os.makedirs(os.path.dirname(savepath), exist_ok=True)
    #     plt.tight_layout()

    #     file, ext = os.path.splitext(savepath)
    #     fig.savefig(file + "_globe" + ext, transparent=True, bbox_inches="tight", pad_inches=0)

    ###fig = draw_map(y, plot_points, title, bds=bds)
    plt.figure(figsize=(15, 10))
    plt.imshow(Y, cmap='viridis')

    #if show or savepath is None:
    #    plt.show()
    #plt.show()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(savepath, format='png', bbox_inches="tight")

def plot_predictions_at_points(spatialencoder, 
                               lonlats, 
                               #degrees_per_pixel = 1,
                               title=None, 
                               show=True, 
                               savepath=None, 
                               class_idx=None,
                               plot_kwargs={},
                               lonlatscrs="4326",
                               plot_crs="4326",
                               ):
    """
    convenience function to plot a scatter plot of values at lonlats
    """
    device = spatialencoder.device

    lons, lats = lonlats[0], lonlats[1]

    # ij indexing to xy indexing
    lons, lats = lons.T, lats.T

    lonlats = torch.Tensor(lonlats)

    with torch.no_grad():
        if spatialencoder.regression:
            Y = spatialencoder(lonlats)
        else:
            Y = torch.sigmoid(spatialencoder(lonlats))

    if class_idx is not None:
        Y = Y[:, class_idx].unsqueeze(-1)

    # if not binary show predictions instead of probabilities
    if not Y.size(-1) == 1:
        y = Y.argmax(-1)
    else:
        y = Y

    # geopandas dataframe of points
    df = pd.DataFrame(lonlats, columns=['longitude', 'latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=lonlatscrs)
    # add predictions
    gdf['y'] = y
    
    if plot_crs != lonlatscrs:
        plot_gdf = gdf.to_crs(epsg=plot_crs) 
    else:
        plot_gdf = gdf
    
    fig = scatter_plot_gdf(plot_gdf, plot_key='y', plot_kwargs=plot_kwargs, title=title)

    if show or savepath is None:
        plt.show()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.tight_layout()
        fig.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)
        
def scatter_plot_gdf(gdf, plot_key, plot_map=True, ax=None, plot_kwargs={}, title=''):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8,8))  
        
    # plot data
    p = gdf.plot(plot_key, ax=ax, **plot_kwargs)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if plot_map: 
        #  coastlines = gpd.read_file(os.path.join(DATA_ROOT,'ne_10m_coastline.shp')  )
        coastlines = gpd.read_file("data/ne_50m_coastline/ne_50m_coastline.shp")  
        coastlines_this_crs = coastlines.to_crs(gdf.crs)  
        coastlines_this_crs.plot(ax=ax, edgecolor='black', linewidth=0.5)  

    # Set the extent of the plot to cover the Arctic area  
    ax.set_xlim(*xlim);  
    ax.set_ylim(*ylim); 

    ax.set_title(title)
    ax.axis("off")
    
    return fig
    
    
def draw_map(y, plot_points, title, bds=[-180,-90,180,90]):
    fig = plt.figure()
    map = Basemap(*bds)
    ax = plt.gca()

    # ax.scatter(points.theta.apply(np.rad2deg), points.phi.apply(np.rad2deg), c=points.land, s=2)
    map.imshow(y.cpu().detach().numpy(), origin="lower", interpolation="none", cmap="RdBu_r", vmin=0, vmax=1)

    if plot_points is not None:
        map.scatter(plot_points[:,0], plot_points[:,1], c="red")

    #map.drawcoastlines()
    map.readshapefile("data/ne_50m_coastline/ne_50m_coastline", "coastlines")
    
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if title is not None:
        ax.set_title(title)

    return fig


def draw_globe(y, lonlats, plot_points, title):

    y = y.squeeze()

    fig = plt.figure()
    map = Basemap(projection='ortho', lat_0=45, lon_0=30, resolution='l')
    ax = plt.gca()
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    # map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral', lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    # map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.
    map.drawmeridians(np.arange(0, 360, 30))
    map.drawparallels(np.arange(-90, 90, 30))

    lons, lats = lonlats.T
    values = y.detach().numpy()  # .reshape(-1)

    x, y = map(lons.numpy() % 360, lats.numpy())

    #map.contourf(x.reshape(180, 360), y.reshape(180, 360), values, cmap="RdBu_r", alpha=.75)
    #map.contour(x.reshape(180, 360), y.reshape(180, 360), values, colors="white", alpha=.5)
    
    if plot_points is not None:
        plot_x, plot_y = map(plot_points[:, 0].numpy() % 360, plot_points[:, 1])
        map.scatter(plot_x, plot_y, c="red")

    if title is not None:
        ax.set_title(title)

    return fig

def find_matrix_plot_filename(resultsdir, pe, nn):
    candidates = os.listdir(resultsdir)
    candidates = [f for f in candidates if (f"{pe:1.8}-{nn:1.6}" in f)]
    candidates = [f for f in candidates if ("globe" not in f)]
    candidates = [f for f in candidates if f.endswith('.png')]
    
    if len(candidates) > 1:
        print('found multiple pngs that fit the criteria for plotting:')
        print(candidates)
    elif len(candidates) == 0:
        print(f'could not find a png that fits the crieteria for plotting', end='')
        print(f' for {pe}, {nn} in {resultsdir}')
        return
    return candidates[0]


def plot_result_matrix(resultsdir, positional_encoders, neural_networks, show=False, savepath=None):
    fig, axs_arr = plt.subplots(len(positional_encoders), len(neural_networks), figsize=(16*len(neural_networks), 10*len(positional_encoders)))
    
    if len(positional_encoders) == 1:
        axs_arr = [axs_arr]
    if len(neural_networks) == 1:
        axs_arr = [axs_arr]
        
    for pe, ax_row in zip(positional_encoders, axs_arr):
        for nn, ax in zip(neural_networks, ax_row):
            filename = find_matrix_plot_filename(resultsdir, pe, nn)

            image = plt.imread(os.path.join(resultsdir,filename))
            ax.imshow(image)
            ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0, transparent=True)