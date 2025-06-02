"""
File containing the functions to load the ERA5 dataset, create geographical maps, and plot the geopotential height or its anomalies over the Euro-Atlantic region.

Author: Adrien Loiseau
Creation date: 30/10/2024
Last modified: 18/05/2025
"""

### Importing the required libraries ###
import time
import scipy
import matplotlib
import numpy as np
import pandas as pd
import netCDF4 as nc
from cftime import num2date
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation

import seaborn as sns
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, to_tree, cut_tree, linkage, fcluster
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_mutual_info_score, calinski_harabasz_score, davies_bouldin_score

### Imports needed for omitted methods ###
# from umap import UMAP
# from sklearn.manifold import TSNE, MDS, SpectralEmbedding, Isomap
# from sklearn.decomposition import FactorAnalysis, KernelPCA, FastICA, NMF
# from sklearn.cluster import MeanShift, SpectralClustering, DBSCAN, HDBSCAN, OPTICS, cluster_optics_dbscan



def load_dataset(file_name):
    """
    Load the ERA5 dataset from a netCDF file and extract the latitude, longitude, geopotential height, and time variables.

    Parameters
    ----------
        - file_name (str): The name of the netCDF file containing the ERA5 dataset.

    Returns
    -------
        - lats (numpy.ndarray): The latitude values from the dataset.
        - longs (numpy.ndarray): The longitude values from the dataset.
        - geopotential_height (numpy.ndarray): The geopotential height values from the dataset.
        - dates (numpy.ndarray): The time values from the dataset.
        - first_year (int): The first year of the dataset.
        - last_year  (int): The last year of the dataset.
    """
    ### Sources ###
    # https://medium.com/analytics-vidhya/how-to-read-and-visualize-netcdf-nc-geospatial-files-using-python-6c2ac8907c7c
    # https://unidata.github.io/netcdf4-python/

    start_time = time.time()

    ### Load the geopotential dataset with netCDF4 ###
    dataset = nc.Dataset(filename=file_name, mode='r')

    ### Extract the latitude and longitude variables from the dataset ###
    lats = dataset.variables['latitude'][:][::2] # Step of 0.5 degrees instead of 0.25 degrees
    longs = dataset.variables['longitude'][:][::2] # Step of 0.5 degrees instead of 0.25 degrees

    ### Extract the geopotential variable from the dataset ###
    geopotential = dataset.variables['z'][:] # Units: m^2 s^-2

    ### Extract the dimensions of the geopotential variable (nb_days, nb_levels, nb_lats, nb_longs) ###
    nb_days, nb_levels, nb_lats, nb_longs = geopotential.shape
    print(f"Geopotential height dataset (0.25°x0.25°): (nb_days={nb_days}, nb_lats={nb_lats}, nb_longs={nb_longs}, grid_size={nb_lats*nb_longs})")

    ### Remove the level dimension as we are only interested in the 500 hPa level ###
    geopotential = geopotential.reshape(nb_days, nb_lats, nb_longs)

    new_geopotential = []
    for day in range(len(geopotential)):
        new_geopotential.append([])
        for row in range(0, len(geopotential[day]), 2):
            new_geopotential[day].append(geopotential[day][row][::2])
    geopotential = np.array(new_geopotential)

    ### Extract the dimensions of the geopotential variable (nb_days, nb_levels, nb_lats, nb_longs) ###
    nb_days, nb_lats, nb_longs = geopotential.shape
    print(f"Geopotential height dataset (0.50°x0.50°): (nb_days={nb_days}, nb_lats={nb_lats}, nb_longs={nb_longs}, grid_size={nb_lats*nb_longs})")
    
    ### Convert geopotential to geopotential height by dividing the geopential by the Earth's gravitational acceleration, g
    geopotential_height = geopotential /scipy.constants.g # Units: m
    print(f"Geopotential height data: {geopotential_height.shape}")

    ### Reshape the 3-D geopotential height array to a 2-D array by combining the latitude and longitude dimensions into one
    geopotential_height_data_reshaped = geopotential_height.reshape(nb_days, nb_lats * nb_longs)
    print(f"Reshaped geopotential height data: {geopotential_height_data_reshaped.shape}")

    ### Extract the time variable from the dataset ###
    times = dataset.variables['valid_time']
    ### Convert the valid_time variable to datetime objects using the num2date function 
    ### (Source: https://unidata.github.io/netcdf4-python/#dealing-with-time-coordinates)
    dates = num2date(times[:], units=times.units, calendar=times.calendar)

    ### Extract the first and last year of the dataset
    first_year = dates[0].year
    last_year = dates[-1].year

    ### Compute the geopotential height average over the entire period ###
    geopotential_height_overall_mean = np.mean(geopotential_height, axis=0)
    geopotential_height_overall_variance = np.var(geopotential_height, axis=0)

    ### Compute the anomalies of the geopotential height values by subtracting the overall average from the daily values
    # The geopotential height anomalies are calculated by subtracting the mean geopotential height over the entire dataset (winter months) from the geopotential height at each grid point.
    geopotential_height_anomalies = geopotential_height - geopotential_height_overall_mean
    print(f"Geopotential height anomalies: {geopotential_height_anomalies.shape}")

    # geopotential_height_anomalies_variance = np.var(geopotential_height_anomalies, axis=0)

    ### Reshape the 3-D geopotential height anomalies array to a 2-D array by combining the latitude and longitude dimensions into one
    geopotential_height_anomalies_reshaped = geopotential_height_anomalies.reshape(nb_days, nb_lats * nb_longs)
    print(f"Reshaped geopotential height anomalies: {geopotential_height_anomalies_reshaped.shape}")

    end_time = time.time()
    print(f"Daily winter data from {first_year} to {last_year} successfully loaded. Loading time: {end_time-start_time:.2f} seconds.")

    # "geopotential_height_overall_variance":geopotential_height_overall_variance, - "geopotential_height_anomalies_variance":geopotential_height_anomalies_variance,
    dataset_dict = {"lats":lats, "longs":longs, "dates":dates, "first_year":first_year, "last_year":last_year, "area":"Euro_Atlantic", \
    "geopotential_height_overall_mean":geopotential_height_overall_mean, \
    "geopotential_height_overall_variance":geopotential_height_overall_variance, \
    "geopotential_height":geopotential_height, "geopotential_height_data_reshaped":geopotential_height_data_reshaped, \
    "geopotential_height_anomalies":geopotential_height_anomalies, "geopotential_height_anomalies_reshaped":geopotential_height_anomalies_reshaped}

    return dataset_dict





def create_geographical_map(ax, area, label_meridians_parallels=False): # labels_meridian_right=False
    """
    Create polar stereographic Basemap instance

    Parameters
    ----------
        - ax (matplotlib.axes._axes.Axes): The axes instance to create the map on.
        - area (str): The area to create the map for. It can be either "Europe_small", "Europe_medium", "Europe_large", or "North_hemisphere".
        - label_meridians_parallels (bool): Whether to display the labels of the meridians and parallels on the sides of the map.
    
    Returns
    -------
        - m (mpl_toolkits.basemap.Basemap): The Basemap instance for the given area.
    """
    ### Set the map projection region ###
    # The map projection region is specified by setting the width and height of the map (in meters),
    # centered on the latitude/longitude specified by lat_0 and lon_0 (in degrees).
    if (area == "Euro_Atlantic"):
        lat_0 = 53 ; lon_0 = -10
        width = 15.5e6 ; height = 7.75e6
    else: # North hemisphere
        width=1e7 ; height=5e6
        lat_0 = 45 ; lon_0 = 0

    ### Create the Basemap instance ###
    # Use the stereographic projection (stere) for the map.
    # Resolution of boundary database to use: c (crude), l (low), i (intermediate), h (high), f (full) or None (no boundaries).
    m = Basemap(projection='stere', resolution='c', \
                width=width, height=height, \
                lat_0=lat_0, lon_0=lon_0, \
                ax=ax)

    ### Draw coastlines and country boundaries ###
    m.drawcoastlines()
    m.drawcountries()

    ### Draw parallels and meridians ###
    if (label_meridians_parallels == True):
        labels_parallel = [0,0,0,1] ; labels_meridian = [1,1,1,0]
    else:
        labels_parallel = [] ; labels_meridian = []

    parallels = np.arange(-90, 91, 10)
    m.drawparallels(parallels, labels=labels_meridian, fontsize=10) # Draw the latitude labels on the left and right hand side of the map

    meridians = np.arange(-180, 181, 20)
    m.drawmeridians(meridians, labels=labels_parallel, fontsize=10) # Draw the longitude labels on the top and bottom of the map

    ### Draw the Arctic Circle at 66.34°N and the Tropic of Cancer at 23.4368°N ###
    # (Source: https://en.wikipedia.org/wiki/Arctic_Circle and https://www.worldatlas.com/articles/where-is-the-arctic-circle.html)
    arctic_circle_lat = 66.34
    troppic_of_cancer_lat = 23.4368
    m.drawparallels([troppic_of_cancer_lat, arctic_circle_lat], color='black', linewidth=1.5, linestyle='--', latmax=arctic_circle_lat)

    ### Needed to see the map boundary in the Jupyter notebook ###
    m.drawmapboundary(color="k", linewidth=1, fill_color=None, ax=ax) # Draw a line around the map region

    return m





def create_contour_data(ax, dataset_dict, type_plot, label_meridians_parallels=False, date_idx=None, geopotential_height_data_to_plot=None, lats=None, longs=None, weather_regime=False):
    """
    Plot the geopotential height or its anomalies over Europe for a specific date if the date index is provided or 
    plot the given geopotential height data if the geopotential_height_data_to_plot is provided.

    Parameters
    ----------
        - ax (matplotlib.axes._axes.Axes): The axes instance to create the contour plot on.
        - dataset_dict (dict): A dictionary containing the dataset and its metadata.
        - type_plot (str): The type of plot to create. It can be either "geopotential_height", "geopotential_height_anomalies_1_day", "geopotential_height_anomalies_temporal_mean", or "principal_component".
        - geopotential_height_data_to_plot (numpy.ndarray): The geopotential height data to plot. It is used if the date index is not provided.

    Returns
    -------
        - m (mpl_toolkits.basemap.Basemap): The Basemap instance for the given area.
        - contour (matplotlib.contour.QuadContourSet): The contour plot of the geopotential height or its anomalies.
        - ticks (numpy.ndarray): The ticks for the colorbar.
    """
    if (lats is None and longs is None): # If the latitude and longitude values are not provided, extract them from the dataset dictionary
        # Extract the latitude, longitude, and area variables from the dataset dictionary.
        lats = dataset_dict["lats"] ; longs = dataset_dict["longs"]
    area = dataset_dict["area"]

    ### Create the geographical map ###
    m = create_geographical_map(ax=ax, area=area, label_meridians_parallels=label_meridians_parallels)

    lon, lat = np.meshgrid(longs, lats) # This converts the latitude/longitude coordinates into a 2D array

    ### Define the color levels, i.e., the range of values of the geopotential height per color
    num_levels = 101 # Number of levels to use in the colorbar
    if (type_plot == "geopotential_height" or type_plot == "mean_geopotential_height"):
        min_val, max_val = 5000, 6000 ; ticks_step = 250
    elif (type_plot == "geopotential_height_anomalies_1_day"):
        min_val, max_val = -300, 300 ; ticks_step = 100
    elif (type_plot == "geopotential_height_anomalies_temporal_mean"):
        min_val, max_val = -200, 200 ; ticks_step = 50
    elif (type_plot == "principal_component"):
        min_val, max_val = -0.017, 0.017 ; ticks_step = 0.0085
    elif (type_plot == "principal_component2"):
        min_val, max_val = -0.1, 0.1 ; ticks_step = 0.05
    elif (type_plot == "variance_geopotential_height" or type_plot == "variance_geopotential_height_anomalies"):
        min_val, max_val = 0, 40000 ; ticks_step = 10000
    elif (type_plot == "2m_temperature"):
        min_val, max_val = -30, +30 ; ticks_step = 5
    elif (type_plot == "2m_temperature_anomalies"):
        min_val, max_val = -15, +15 ; ticks_step = 3
    elif (type_plot == "DBSCAN"):
        min_val, max_val = -0.5, 0.5 ; ticks_step = 0.25
    elif (type_plot == "spectral_clustering"):
        min_val, max_val = -10, 10 ; ticks_step = 2.5
    elif (type_plot == "significance"):
        min_val, max_val = 0, 0.0015 ; ticks_step = 0.0005
    elif (type_plot == "mean_geopotential_height_anomalies"):
        min_val, max_val = -50, 50 ; ticks_step = 25

    levels = np.linspace(min_val, max_val, num_levels)
    ticks = np.arange(min_val, max_val+ticks_step, ticks_step) # Define the ticks for the colorbar

    ### Extract the geopotential height data to plot
    if (date_idx != None):
        if (type_plot == "geopotential_height"):
            geopotential_height_data_to_plot = dataset_dict["geopotential_height"][date_idx]
        elif ((type_plot == "geopotential_height_anomalies_1_day") or (type_plot == "geopotential_height_anomalies_temporal_mean")):
            geopotential_height_data_to_plot = dataset_dict["geopotential_height_anomalies"][date_idx]

    ### Draw filled contours representing the geopotential height
    # extend='both' means that the colorbar will have extensions on both ends of the colorbar, which will color the values below and above the levels range.
    # latlon=True, x,y are intrepreted as longitude and latitude in degrees:
    # Data and longitudes are automatically shifted to match the map projection region; x,y are transformed to map projection coordinates.
    if (type_plot == "variance_geopotential_height" or type_plot == "variance_geopotential_height_anomalies" or type_plot == "significance"):
        cmap = "binary" # Greys
    elif (type_plot == "geopotential_height" or type_plot == "mean_geopotential_height"):
        cmap = "coolwarm" # jet coolwarm
    else:
        cmap = "PuOr_r"
    contour = m.contourf(x=lon, y=lat, data=geopotential_height_data_to_plot, levels=levels, cmap=cmap, extend='both', latlon=True)

    ### Plot the contour lines of the geopotential height ###
    if (weather_regime == True):
        # print(np.min(geopotential_height_data_to_plot + dataset_dict["geopotential_height_overall_mean"]), np.max(geopotential_height_data_to_plot + dataset_dict["geopotential_height_overall_mean"]))
        CS = m.contour(x=lon, y=lat, data=geopotential_height_data_to_plot + dataset_dict["geopotential_height_overall_mean"], levels=np.arange(4900, 6101, 100), colors='k', linewidths=1, latlon=True)
        ax.clabel(CS, levels=np.arange(5000, 6001, 200), inline=True, fontsize=11, fmt='%1.0f', colors='k') # Label the contour lines with their values
    if (type_plot == "mean_geopotential_height"):
        # print(np.min(geopotential_height_data_to_plot), np.max(geopotential_height_data_to_plot))
        CS = m.contour(x=lon, y=lat, data=geopotential_height_data_to_plot, levels=np.arange(4900, 6101, 100), colors='k', linewidths=1, latlon=True)
        ax.clabel(CS, levels=np.arange(5000, 6001, 200), inline=True, fontsize=11, fmt='%1.0f', colors='k')

    return m, contour, ticks





def compute_date_idx(day, month, year, first_year, last_year):
    """
    Compute the index of the date in the dataset.

    Parameters
    ----------
        - day (int): The day of the date. Must be within the range of dates of the month.
        - month (int): The month of the date. Must be either 1 (January), 2 (February), or 12 (December).
        - year (int): The year of the date. Must be within the range of the dataset.
        - first_year (int): The first year of the dataset.
        - last_year (int): The last year of the dataset.

    Returns
    -------
        - date_idx (int): The index of the date in the dataset.
    """
    ### Check if the given date is correct ###
    if (year < first_year or year > last_year):
        print(f"The year ({year}) must be within the range of the dataset ({first_year}-{last_year}).")
        return
    elif (month not in [1, 2, 12]):
        print(f"The month ({month}) must be either 'December', 'January' or 'February'.")
        return
    elif (month == 1 or month == 12): # January or December
        if (day < 1 or day > 31):
            print(f"The day ({day}) must be within the range of the month (1-31).")
            return
    else: # February
        if (year % 4 != 0): # Non-leap year
            if (day < 1 or day > 28):
                print(f"The day ({day}) must be within the range of the month (1-28).")
                return
        else: # Leap year
            if (day < 1 or day > 29):
                print(f"The day ({day}) must be within the range of the month (1-29).")
                return

    year_start_idx = 0

    for year_ in range(first_year, year):
        if (year_ % 4 != 0): # Non-leap year
            year_start_idx += 90
        else: # Leap year
            year_start_idx += 91

    if (month == 1): # January
            date_idx = year_start_idx + day - 1
    elif (month == 2): # February
        date_idx = year_start_idx + 31 + day - 1
    else: # December
        if (year % 4 != 0): # Non-leap year
            date_idx = year_start_idx + 31 + 28 + day - 1
        else: # Leap year
            date_idx = year_start_idx + 31 + 29 + day - 1

    return date_idx





def plot_geopotential_height(dataset_dict, type_plot, date=None, set_title=True, save_fig=False, show_fig=True):
    """
    Plot the geopotential height or its anomalies over Europe for a specific date.

    Parameters
    ----------
        - dataset_dict (dict): A dictionary containing the dataset and its metadata.
        - date (str): The date to plot the geopotential height for. The date must be in the format "YYYY_MM_DD".
        - type_plot (str): The type of plot to create. It can be either "geopotential_height", "geopotential_height_anomalies_1_day", "geopotential_height_anomalies_temporal_mean", or "principal_component".
        - save_fig (bool): Whether to save the figure as an image file or not.
        - show_fig (bool): Whether to display the figure or not.
    """
    ### Create figure and axes instances ###
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = plt.gca() # Get the current axes or create one if there are no current axes

    dates = dataset_dict['dates']

    if (date != None):
        date_components = date.split("_")
        date_idx = compute_date_idx(year=int(date_components[0]), month=int(date_components[1]), day=int(date_components[2]), \
                                            first_year=dataset_dict["first_year"], last_year=dataset_dict["last_year"])
        ### Create the contour plot ###
        m, contour, ticks = create_contour_data(ax=ax, dataset_dict=dataset_dict, type_plot=type_plot, date_idx=date_idx)
    else:
        if (type_plot == "variance_geopotential_height_anomalies"):
            geopotential_height_data_to_plot = dataset_dict["geopotential_height_anomalies_variance"]
        elif (type_plot == "mean_geopotential_height"):
            geopotential_height_data_to_plot = dataset_dict["geopotential_height_overall_mean"]
        elif (type_plot == "variance_geopotential_height"):
            geopotential_height_data_to_plot = dataset_dict["geopotential_height_overall_variance"]

        ### Create the contour plot ###
        m, contour, ticks = create_contour_data(ax=ax, dataset_dict=dataset_dict, type_plot=type_plot, geopotential_height_data_to_plot=geopotential_height_data_to_plot)
    
    ### Add a colorbar ###
    cbar = m.colorbar(contour, ax=ax, ticks=ticks, location='bottom', pad="5%") # 10%
    # cbar = fig.colorbar(contour, ax=ax, ticks=ticks, location='bottom', pad=0.02, fraction=0.05) # 5%
    # if ((type_plot == "geopotential_height") or (type_plot == "mean_geopotential_height") or (type_plot == "variance_geopotential_height") or (type_plot == "variance_geopotential_height_anomalies")):
    #     # cbar.set_label(r'500 hPa geopotential height $[m]$')
    #     cbar.ax.text(x=1.1, y=-1.5, s=r"$[m]$", transform=cbar.ax.transAxes, ha='center', va='center', fontsize=10)
    # elif ((type_plot == "geopotential_height_anomalies_1_day") or (type_plot == "geopotential_height_anomalies_temporal_mean")): 
    #     cbar.set_label(r'500 hPa geopotential height anomalies $[m]$')
    # elif (type_plot == "principal_component"):
    #     cb.set_label(r'[-]', loc='right') # Unitless

    if (type_plot == "principal_component"):
        units = r"$[-]$"
    else:
        units = r"$[m]$"
    # cbar.ax.text(x=1.15, y=-1.5, s=units, transform=cbar.ax.transAxes, ha='center', va='center', fontsize=10)
    cbar.ax.tick_params(axis="x", labelsize=12)

    if (set_title == True):
        if (date != None):
            if (type_plot == "geopotential_height"): # Geopotential height on a specific date
                ax.set_title(f"500 hPa Geopotential height \non {dates[date_idx]}")
            elif (type_plot == "geopotential_height_anomalies_1_day"): # On a specific date
                ax.set_title(f"500 hPa Geopotential height anomalies \non {dates[date_idx]}")
            elif (type_plot == "geopotential_height_anomalies_temporal_mean"): # During a period of time (month, season, year, multiple years)
                ax.set_title(f"500 hPa Geopotential height anomalies \nduring {dates[date_idx]}")
        else:
            if (type_plot == "variance_geopotential_height_anomalies"):
                ax.set_title(f"Variance of the 500 hPa geopotential height anomalies\nover the winters (DJF) from {dataset_dict["first_year"]} to {dataset_dict["last_year"]}")
            elif (type_plot == "mean_geopotential_height"):
                ax.set_title(f"Mean of the 500 hPa geopotential height values\nover the winters (DJF) from {dataset_dict["first_year"]} to {dataset_dict["last_year"]}")
            elif (type_plot == "variance_geopotential_height"):
                ax.set_title(f"Variance of the 500 hPa geopotential height values\nover the winters (DJF) from {dataset_dict["first_year"]} to {dataset_dict["last_year"]}")

    if (date != None):
        if (type_plot == "geopotential_height"):
            file_name = f"{type_plot}_{dates[date_idx].year}_{dates[date_idx].month:02d}_{dates[date_idx].day:02d}"
        else:
            file_name = f"{type_plot}_{dates[date_idx].year}_{dates[date_idx].month:02d}_{dates[date_idx].day:02d}_reference_period_{dataset_dict["first_year"]}_{dataset_dict["last_year"]}"
    else:
        file_name = f"{type_plot}_{dataset_dict["first_year"]}_{dataset_dict["last_year"]}"

    if (save_fig == True): plt.savefig(f"Images/{file_name}.png", bbox_inches='tight', dpi=300)
    if (show_fig == True): plt.show()
    else: return fig, ax, contour






def retrieve_temperature_data_for_clusters(y_pred, nb_clusters, dataset_dict):
    ### Daily statistics of the 2-m air temperature at Brussels (data ERA5) ###
    df = pd.read_csv("Data/dailyStatistics_T2m_Bruxelles.csv")

    winter_year = 1959 # Start year of the data
    for year in range(dataset_dict["first_year"], winter_year+1):
        if (year == dataset_dict["first_year"]): year_start_idx = 0 ; year_end_idx = 60
        elif (year % 4 == 0): year_start_idx = year_end_idx ; year_end_idx += 91 # Leap year
        else: year_start_idx = year_end_idx ; year_end_idx += 90 # Non-leap year
        first_temperature_date_idx = year_start_idx + 31 # --> 01/01/1959

    winter_date_idx = 0
    avg_temperature = np.zeros((3, nb_clusters))
    min_temperature = np.zeros((3, nb_clusters))
    max_temperature = np.zeros((3, nb_clusters))
    number_of_days_with_temperature = np.zeros((3, nb_clusters))

    for date_idx in range(df[df.columns[0]].shape[0]):
        date  = df[df.columns[0]][date_idx]
        date_components = date.split("-")
        year = int(date_components[0]) ; month = int(date_components[1]) ; day = int(date_components[2])

        if (month in [12, 1, 2]): # Keep only winter months
            month_idx = np.argwhere(np.array([12, 1, 2]) == month)[0][0] # Month index in the winter season

            winter_date_idx += 1
            cluster_idx = y_pred[first_temperature_date_idx + winter_date_idx]
            number_of_days_with_temperature[month_idx, cluster_idx] += 1

            avg_temperature[month_idx, cluster_idx] += df[df.columns[1]][date_idx]
            min_temperature[month_idx, cluster_idx] += df[df.columns[2]][date_idx]
            max_temperature[month_idx, cluster_idx] += df[df.columns[3]][date_idx]

    avg_temperature = avg_temperature / number_of_days_with_temperature
    min_temperature = min_temperature / number_of_days_with_temperature
    max_temperature = max_temperature / number_of_days_with_temperature

    return avg_temperature, min_temperature, max_temperature, number_of_days_with_temperature