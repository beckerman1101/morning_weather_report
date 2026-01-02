#### ---- NECESSARY PYTHON PACKAGES ---- ####

# All of them should import, but if any errors arise, just add the missing packages
# to the cell above after cfgrib and re-run the notebook
# Most of them should be available within Colab's base environment



import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature,NaturalEarthFeature
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import cfgrib
from scipy.spatial import cKDTree
import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import requests
from zoneinfo import ZoneInfo
import zipfile
import os
import io
from io import BytesIO
from PIL import Image
import gzip
import shutil
import tarfile
from bs4 import BeautifulSoup
import re
import textwrap
from skimage.transform import resize
from awips.dataaccess import DataAccessLayer
from scipy.interpolate import griddata
from github import Github


# These strings are how every export is labeled, as well as the title

today = datetime.now(ZoneInfo('America/Denver'))
yesterday = today - timedelta(days=1)
todayst = today.strftime('%a %m/%d')
yesterdayst = yesterday.strftime('%a %m/%d')
yesterdaystr = yesterday.strftime('%Y%m%d')
todaystr = today.strftime('%Y%m%d')
mo = today.strftime('%Y%m')
ts = today.strftime('%I:%M %p')

# GitHub Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  
REPO_NAME = 'beckerman1101/daily_accum_mapping'  
BRANCH_NAME = 'main'



os.environ['SHAPE_RESTORE_SHX'] = 'YES'
base_dir = os.path.dirname(os.path.abspath(__file__))


"""### Installing CDOT Trebuchet Font, setting up variables for today (for use in labels & filemaking)"""

#### ---- FONT SETUP ---- ####

# Not sure if adding all of them is necessary because the plot looks good, but not taking chances here

treb = os.path.join(base_dir, 'TREBUC.TTF')
trebb = os.path.join(base_dir, 'TREBUCBD.TTF')
trebi = os.path.join(base_dir, 'TREBUCIT.TTF')
trebbi = os.path.join(base_dir, 'TREBUCBI.TTF')
fm.fontManager.addfont(treb)
fm.fontManager.addfont(trebb)
fm.fontManager.addfont(trebi)
fm.fontManager.addfont(trebbi)
treb = fm.FontProperties(fname=treb)
trebb = fm.FontProperties(fname=trebb)
trebi = fm.FontProperties(fname=trebi)
trebbi = fm.FontProperties(fname=trebbi)

#### ---- DATE STRINGS ---- ####



"""### Importing Highway logos, establishing Colorado lat/lon bounds, importing and processing necessary roadway/CDOT shapefiles"""

#### ---- LOGO & SHAPEFILE IMPORTS ---- ####

# These open up the highway logos and cdot for the top of the figure. The cdot logo is a mess, see the final few lines of the notebook

cdotlogo = mpimg.imread(os.path.join(base_dir, 'images', 'image (1).png'))
nws = mpimg.imread(os.path.join(base_dir, 'images', 'nwslogo.png'))
us160 = mpimg.imread(os.path.join(base_dir, 'images', 'US_160.png'))
us550 = mpimg.imread(os.path.join(base_dir, 'images', 'US_550.png'))
us50 = mpimg.imread(os.path.join(base_dir, 'images', 'US_50.png'))
us40 = mpimg.imread(os.path.join(base_dir, 'images', 'US_40.png'))
us285 = mpimg.imread(os.path.join(base_dir, 'images', 'US_285.png'))
i25 = mpimg.imread(os.path.join(base_dir, 'images', 'I-25.png'))
i70 = mpimg.imread(os.path.join(base_dir, 'images', 'I-70.png'))
i76 = mpimg.imread(os.path.join(base_dir, 'images', 'I-76.png'))
size = (500, 500, 4)
nws = resize(nws, size, anti_aliasing=True)

# Colorado bounds and padding for the maps. This can be adjusted if we want to scale the mapping
co_bounds = [-109.05,-102.05,37,41]
x = .15

# Placefiles for counties, interstates, state highways, and manipulating the regions shapefile to plot properly
# My laziness backfired on the cdot boundaries. I ignored the projection file and had to manually manipulate it, but this works
# Don't fix what ain't broken

counties = gpd.read_file(os.path.join(base_dir, 'shapefiles', 'tl_2019_08_county.shp'))
cdot_path = os.path.join(base_dir, 'shapefiles', 'MaintenanceSections.shp')
cdot = gpd.read_file(cdot_path).set_crs(epsg=26913)
cdot = cdot.to_crs(epsg=32662)
cdot_bounds = cdot.total_bounds
x_scale = (co_bounds[1]-co_bounds[0])/(cdot_bounds[2]-cdot_bounds[0])
y_scale = (co_bounds[3]-co_bounds[2])/(cdot_bounds[3]-cdot_bounds[1])
cdot = cdot.scale(x_scale, y_scale, origin=(0,0))
cdot_bounds2 = cdot.total_bounds
lx = cdot_bounds2[0]
ly = cdot_bounds2[1]
cdot = cdot.translate(xoff=co_bounds[0]-lx, yoff=co_bounds[2]-ly)
sh = gpd.read_file(os.path.join(base_dir, 'shapefiles', 'tl_2021_08_prisecroads.shp'))
interstates = gpd.read_file(os.path.join(base_dir, 'shapefiles', 'us_interstate_highways.shp'))

"""### Establishing colorbars and their labels for snowfall and WWA Display"""

#### ---- CREATING COLORBARS ---- ####

# Snowfall colorbar
colors_snow = [
    '#FFFFFF', '#E7E7E7', '#CBE0ED', '#89BFDF', '#5A9CCB', '#3675B2', '#3650AB',
    '#FFFFAD', '#FFD130', '#FFA030', '#E34130', '#B23030', '#883030', '#D7D7FF', '#B3A4E1',
    '#9775B8', '#7B458F', '#5A305B']
cmaps = mcolors.ListedColormap(colors_snow)
boundariess = [0,0.1,1,2,3,4,5,6,8,10,12,14,16,18,24,30,36,42]
norms = mcolors.BoundaryNorm(boundariess, cmaps.N)
snow_labels = ['','0.1"','1"','2"','3"','4"','5"','6"','8"','10"','12"','14"','16"','18"','24"','30"','36"','']

#WWA colorbar
wwa_colors = {
    "Red Flag Warning": "#F60093",
    "High Wind Warning": "#D6A330",
    "Blizzard Warning": "#F73B19",
    "Winter Weather Advisory": '#7B6AEB',
    "Winter Storm Warning": "#F865B3",
    "Winter Storm Watch": '#4E83B2'
    # Add other warnings and advisories as needed, grab the hexadecimal of the color from NWS
}
cmapw = mcolors.ListedColormap(list(wwa_colors.values()))
normw = mcolors.BoundaryNorm(np.arange(0, len(wwa_colors)+1), len(wwa_colors))

# Black colorbar

black_cmap = ListedColormap([(0, 0, 0, 1)])

"""### Query & process most recent AFD"""

#### ---- AFD QUERY ---- ####

#ChatGPT wrote this whole one except the last line. Generates a pretty little list out of the AFD text though


afd_url = "https://forecast.weather.gov/product.php?site=BOU&issuedby=BOU&product=AFD&format=CI&version=1&glossary=1"
response = requests.get(afd_url, stream=True)
afd_text = response.text
soup = BeautifulSoup(afd_text, "html.parser")
clean_text = soup.get_text()
match = re.search(r"\.KEY MESSAGES\.(.*?)&&", clean_text, re.DOTALL)
key_messages = match.group(1).strip()
key_messages = re.sub(r"\n(?![-*])", " ", key_messages)
key_messages = re.sub(r"\s+", " ", key_messages).strip()
key_messages = re.sub(r'\([^\)]*\)', lambda m: m.group(0).replace('-', '±'), key_messages)
messages_list = re.split(r' - ', key_messages)
messages_list = [msg.strip() for msg in messages_list if msg.strip()]
messages_list = [msg.replace('±', '-') for msg in messages_list]
afdtext = messages_list[1:]

"""### Query & process WWA"""

#### ---- NWS WATCH, WARNING, AND ADVISORY DATA ---- ####

# Some of this filtering is a little overkill, but grabs only the Colorado winter-weather products
# If we want Dense Fog, Wind Chill, we can add them, but... this is enough for this graphic

# All of these queries that are ran download the file to the extracted file folder, we can keep for records

url = 'https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz'
downloaded_file_path = os.path.join(base_dir, f'{todaystr}_wwa.tar.gz')
extracted_folder = base_dir
response = requests.get(url, stream=True)
with open(downloaded_file_path, 'wb') as f:
    f.write(response.content)
print(f"File downloaded: {downloaded_file_path}")
with tarfile.open(downloaded_file_path, 'r:gz') as tar:
    tar.extractall(path=extracted_folder)
    print(f"Extracted files to: {extracted_folder}")
shapefile_path = None
for file in os.listdir(extracted_folder):
    if file.endswith('.shp'):
        shapefile_path = os.path.join(extracted_folder, file)
        break
if shapefile_path is None:
    print("No shapefile found in the extracted files.")
else:
    gdf = gpd.read_file(shapefile_path)
    print(f"Shapefile loaded: {shapefile_path}")

co = gdf[gdf['WFO'].isin(['KBOU' , 'KGJT' , 'KPUB' , 'KGLD'])].copy()
co['Warning Color'] = co['PROD_TYPE'].map(wwa_colors)
co = co[pd.notna(co['Warning Color'])]
co = co.reset_index(drop=True)

#### ---- NWS SNOWFALL DATA QUERIES ---- ####

# The file formats are different and have to be handled as such

# The forecast has to be processed to sum up through the next 12z step of the NDFD

fcst_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.crrocks/VP.001-003/ds.snow.bin"
fcst_name = os.path.join(base_dir, f'{todaystr}_ndfdsnow.bin')
response = requests.get(fcst_url, stream=True)
response.raise_for_status()
with open(fcst_name, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
print(f"Forecast file downloaded: {fcst_name}")
snow_fcst = xr.open_dataset(fcst_name, engine='cfgrib', decode_timedelta=True)
times = snow_fcst.valid_time.values
pos1 = None
count = 0
for dt in times:
    hour = dt.astype('datetime64[h]').astype(int) % 24
    if hour == 12:
        count += 1
        if count == 2:
            pos1 = dt
            break
if pos1 is not None:
    mask = snow_fcst['valid_time'] == pos1
    pos = np.where(mask)[0][0]
else:
    pos = -1
end_slice = pos+1
snow_m = snow_fcst.isel(step=slice(0, end_slice)).unknown.sum(dim='step')
end = snow_fcst.step[pos].valid_time.values.astype('datetime64[s]').astype(datetime).replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Denver")).strftime('%I%p %a')

# List of timestamps to download
ts_list = [yesterdaystr + '12', yesterdaystr + '18', todaystr + '00', todaystr + '06']

# Download and process NOHRSC data
datasets = []
for timestamp in ts_list:
    ts_dt = datetime.strptime(timestamp, "%Y%m%d%H")
    mo_ts = ts_dt.strftime('%Y%m')

    accum_url = (
        f"https://www.nohrsc.noaa.gov/snowfall_v2/data/"
        f"{mo_ts}/sfav2_CONUS_6h_{timestamp}.nc")
    accum_name = f'{ts_dt}_gridded.nc'

    response = requests.get(accum_url, stream=True)
    if response.status_code == 200:
        with open(accum_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {accum_name}")

        # Open the dataset
        ds = xr.open_dataset(accum_name)
        datasets.append(ds)
    else:
        print(f"Failed to download: {accum_url}")

# Combine datasets and sum over time
if datasets:
    snow_accum_nohrsc = xr.concat(datasets, dim="time").sum(dim="time")
    print("Successfully combined and summed NOHRSC snowfall datasets")
    nohrsc = snow_accum_nohrsc
else:
    print("No files were successfully downloaded.")

# NOW REPLACE THE python-awips SECTION WITH THIS:

import rasterio
from pyproj import Transformer

# Construct NBM GeoTIFF URL for 06Z run
nbm_url = f"https://noaa-nbm-pds.s3.amazonaws.com/blendv4.3/conus/{today.year}/{today.strftime('%m')}/{today.strftime('%d')}/0600/snowamt06/blendv4.3_conus_snowamt06_{today.strftime('%Y-%m-%d')}T06%3A00_{today.strftime('%Y-%m-%d')}T12%3A00.tif"

try:
    # Open NBM GeoTIFF
    with rasterio.open(nbm_url) as src:
        nbm_data = src.read(1)
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width
        
        # Build 2D grids of x/y coordinates in the native projection
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten(), offset="center")
        xs = np.array(xs).reshape(h, w)
        ys = np.array(ys).reshape(h, w)
        
        # Transform to lat/lon (EPSG:4326)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        nbm_lons, nbm_lats = transformer.transform(xs, ys)
    
    # Extract the 1D lat/lon from NOHRSC (from nohrsc variable created earlier)
    target_lat = nohrsc['lat'].values
    target_lon = nohrsc['lon'].values
    
    # Create 2D meshgrid for target coordinates
    target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
    
    # Flatten the NBM source data for interpolation
    source_points = np.column_stack([nbm_lons.flatten(), nbm_lats.flatten()])
    source_values = nbm_data.flatten()
    
    # Remove any NaN values
    valid_mask = ~np.isnan(source_values)
    source_points = source_points[valid_mask]
    source_values = source_values[valid_mask]
    
    # Interpolate NBM data onto NOHRSC grid
    print("Regridding NBM data to NOHRSC grid... this may take a moment")
    nbm_regridded = griddata(
        source_points,
        source_values,
        (target_lon_2d, target_lat_2d),
        method='linear',
        fill_value=np.nan
    )
    
    # Create xarray DataArray for regridded NBM
    nbm_regridded_xr = xr.DataArray(
        nbm_regridded,
        dims=["lat", "lon"],
        coords={"lat": target_lat, "lon": target_lon}
    )
    
    # Convert to inches and sum with NOHRSC data
    nbm_inches = nbm_regridded_xr.copy() # Convert meters to inches
    nohrsc_inches = nohrsc['Data'] * 39.3701  # Convert meters to inches
    
    # Sum the datasets
    total_snowfall = nbm_inches + nohrsc_inches
    
    # Filter to Colorado bounds and interpolate
    df_co = total_snowfall.where(
        (total_snowfall.lat >= co_bounds[2]) &
        (total_snowfall.lat <= co_bounds[3]) &
        (total_snowfall.lon >= co_bounds[0]) &
        (total_snowfall.lon <= co_bounds[1]),
        drop=True
    )
    
    # Get the number of latitude and longitude points in the filtered data
    latlen = len(df_co.lat)
    lonlen = len(df_co.lon)
    
    # Define the interpolation factor
    inter_factor = 8
    
    # Create target latitude and longitude arrays
    target_lat_interp = np.linspace(co_bounds[2], co_bounds[3], latlen * inter_factor)
    target_lon_interp = np.linspace(co_bounds[0], co_bounds[1], lonlen * inter_factor)
    
    # Interpolate onto finer grid
    snow_accumulation = df_co.interp(lat=target_lat_interp, lon=target_lon_interp, method='linear')
    
    accum_end = '6am'
    print("Successfully processed NBM GeoTIFF and combined with NOHRSC data")

except Exception as e:
    print(f"Error processing NBM GeoTIFF: {e}")
    print("Falling back to NOHRSC data only")
    # Fallback to NOHRSC only
    df2 = nohrsc
    df_co = df2.where(df2.lat>=co_bounds[2], drop=True).where(df2.lat<=co_bounds[3], drop=True).where(df2.lon>=co_bounds[0], drop=True).where(df2.lon<=co_bounds[1], drop=True)
    latlen = len(df_co.lat)
    lonlen = len(df_co.lon)
    inter_factor = 8
    target_lat = np.linspace(co_bounds[2], co_bounds[3], latlen*inter_factor)
    target_lon = np.linspace(co_bounds[0], co_bounds[1], lonlen*inter_factor)
    snow_accumulation = 39.3701 * df_co['Data'].interp(lat=target_lat, lon=target_lon, method='linear')
    accum_end = '12am'


df1 = 39.3701*snow_m
inter_factor = 8
new_lon = np.linspace(df1.y[0].item(), df1.y[-1].item(), df1.sizes["y"] * inter_factor)
new_lat = np.linspace(df1.x[0].item(), df1.x[-1].item(), df1.sizes["x"] * inter_factor)
snow_forecast = df1.interp(x=new_lat, y=new_lon)
df3 = snow_accumulation

table_file = os.path.join(base_dir, 'fcst_locations.csv')
table = pd.read_csv(table_file)
lats_lons = np.vstack([df1['latitude'].values.ravel(), df1['longitude'].values.ravel()]).T
kdtree = cKDTree(lats_lons)
lats, lons, vals, ranges, fcst= [], [], [], [], []
for i in range(len(table)):
    lat = table.loc[i].lat
    lon = (360 + table.loc[i].lon)
    longi = table.loc[i].lon
    coords = np.array([[lat, lon]])
    dist, index = kdtree.query(coords)
    loc = df1.isel(y=index[0] // df1.sizes['x'], x=index[0] % df1.sizes['x'])
    fcst_val = df3.sel(lat=lat, method='nearest').sel(lon=longi, method='nearest').values.max()
    val = loc.values.max()
    range_val = round(val)
    if range_val==0:
      ranges.append('0"')
    elif range_val<=3:
      buffer = 1
      ranges.append(f'{range_val-buffer}" - {range_val+buffer}"')
    elif range_val>3:
      buffer = 2
      ranges.append(f'{range_val-buffer}" - {range_val+buffer}"')
    lats.append(lat)
    lons.append(longi)
    vals.append(val)
    fcst.append(fcst_val)
log = pd.DataFrame({'lat': lats, 'lon': lons, 'fcst': vals, 'accum':fcst, 'range':ranges}, index=table.location.values)



"""## Generate the Report
#### This will save to today's folder in the Morning Weather Report Drive folder
"""

#### ---- REPORT GENERATION ---- ####

# Initializing the figure, gridspec, and subplots to place information in

fig = plt.figure(figsize=(90.7485, 45), tight_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[0.05, 1.27, 0.15, 0.85], height_ratios=[5, 4])
plt.rcParams['font.family'] = treb.get_name()
ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 2:], projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[1, 0:2], projection=ccrs.PlateCarree())
cax = fig.add_subplot(gs[0, 0])
legend_ax = fig.add_subplot(gs[1, 2])
wwa_ax = fig.add_subplot(gs[1, 3], projection=ccrs.PlateCarree())
ax1pos = ax1.get_position()
ax2pos = ax2.get_position()
ax3pos = ax3.get_position()
caxpos = cax.get_position()
legend_axpos = legend_ax.get_position()
wwa_axpos = wwa_ax.get_position()

# Plotting the snow maps with counties, SHs, interstates, and CDOT sections

ax1.add_feature(cfeature.STATES)
counties.plot(ax=ax1, color='none', ec='gray', alpha=0.75, linewidth=3, zorder=14)
ax1.set_extent([-106.5, -103.5, 39.05, 40.5])
interstates.plot(ax=ax1, ec='r', linewidth=4)
sh.plot(ax=ax1, ec='r', linewidth=1.5, alpha=0.7)
ax1.imshow(i25, extent=[-105.029, -104.929, 40.067, 40.167], zorder=15) #40.11712370587691, -104.97956689965562
ax1.imshow(i70, extent=[-106.163, -106.063, 39.525, 39.625], zorder=15) #39.575197779681986, -106.11314206690821
ax1.imshow(i70, extent=[-104.015, -103.915, 39.445, 39.545], zorder=15) #39.496990632924955, -103.96489627317092
ax1.imshow(i76, extent=[-104.433, -104.333, 40.119, 40.219], zorder=15) #40.169462874547506, -104.38319636781574
ax1.imshow(us40, extent=[-105.856, -105.776, 39.905, 39.985], zorder=15) #39.94835413043858, -105.816784876115
ax1.imshow(us285, extent=[-105.45, -105.34, 39.424, 39.504], zorder=15) #39.464749902186234, -105.39577686384864

ax2.set_extent([co_bounds[0]-x, co_bounds[1]+x, co_bounds[2]-x, co_bounds[3]+x])
ax2.add_feature(cfeature.STATES, ec='k', linewidth=2)
counties.plot(ax=ax2, color='none', ec='gray', alpha=0.75, zorder=20)
sh.plot(ax=ax2, ec='r', linewidth=2, alpha=0.7)
interstates.plot(ax=ax2, ec='r', linewidth=3)
cdot.plot(ax=ax2, aspect=1, fc='none', ec='k', linewidth=4, zorder=12)

ax2.imshow(i25, extent=[-104.90, -104.75, 37.75, 37.9], zorder=15)
ax2.imshow(i25, extent=[-105.06, -104.91, 40.35, 40.5], zorder=15)
ax2.imshow(i70, extent=[-107.20, -107.05, 39.55, 39.7], zorder=15)
ax2.imshow(i70, extent=[-103.75, -103.6, 39.18, 39.33], zorder=15)
ax2.imshow(i76, extent=[-103.6, -103.45, 40.2, 40.35], zorder=15)
ax2.imshow(us40, extent=[-106.23, -106.13, 40, 40.1], zorder=15)
ax2.imshow(us40, extent=[-108, -107.9, 40.46, 40.56], zorder=15)
ax2.imshow(us50, extent=[-107.3, -107.2, 38.42, 38.52], zorder=15)
ax2.imshow(us50, extent=[-104.09, -103.99, 38.08, 38.18], zorder=15)
ax2.imshow(us550, extent=[-107.89, -107.76, 37.45, 37.55], zorder=15)
ax2.imshow(us285, extent=[-105.63, -105.50, 39.4, 39.5], zorder=15)
ax2.imshow(us160, extent=[-106.78, -106.65, 37.55, 37.65], zorder=15)
ax2.imshow(us160, extent=[-103.14, -103.01, 37.22, 37.32], zorder=15)

# Mapping the snow data on the respective maps

accum_map = snow_accumulation.plot(ax=ax1, x='lon', y='lat', transform=ccrs.PlateCarree(),
                              cmap=cmaps, norm=norms, add_colorbar=False)
fcst_map = snow_forecast.plot(ax=ax2, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                              cmap=cmaps, norm=norms, add_colorbar=False)
ax1.set_title(f'Interpolated Snowfall Accumulation: 12am yesterday - 4am today', x=0, fontsize=70, ha='left', style='italic', pad=20)
ax2.set_title(f'Forecasted Snowfall through {end}', x=0, fontsize=70, ha='left', style='italic', pad=20)

# Setting up the snow colorbar

cbar = fig.colorbar(accum_map, cax=cax, orientation='vertical', ticks=boundariess)
cbar.ax.yaxis.set_label_position("left")
cbar.ax.yaxis.tick_left()
cbar.ax.set_yticklabels(snow_labels)
cbar.ax.tick_params(labelsize=50, pad=20)
cax.set_anchor('W')  

# Quick text wrapping function

def wrap_label(text, width=8):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

# Whole lot of messy code for info on the bottom half of the report


ax3.set_xlim(0,50.7485)
ax3.set_ylim(0,20)
ax3.text(1, 17.75, 'Key Messages:', fontsize=80, ha='left')
ax3.text(30, 17.75, '48-Hr Forecasted Snow in High Traffic Areas', fontsize=80, ha='left')
for i in range(len(afdtext)):
    string = wrap_label(afdtext[i], width=64)
    length = len(string.split('\n'))
    if length>=5 and i==0:
        y=12.5
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')
    elif length<=2 and i==0:
        y=15
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')
    elif length>=3 and i==0:
        y=13.25
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')
    if length>=5 and i!=0:
        y-=5.5
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')
    elif length<=2 and i!=0:
        y-=3
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')  
    elif length>=3 and i!=0:
        y-=4
        ax3.text(1, y, f'- {string}', fontsize=50, ha='left', style='italic')
ax3.axis('off')
# Snow Forecast values

# Based on point data, creates a range. If snow is 0<x<=3, the range will be 2" around
# the value, so 2" will produce 1-3"
# If x>3, the range is 4"


ax1.scatter(log['lon'], log['lat'], c=log['accum'], cmap=black_cmap, s=12500, zorder=25, edgecolors='black', transform=ccrs.PlateCarree())
for i, row in log.iterrows():
    ax1.text(row['lon'], row['lat'], f'{int(row["accum"])}"', fontsize=60, ha='center', va='center', zorder=30, color='white', transform=ccrs.PlateCarree(), weight='bold')


y = 15
for i in log.index:
  loc = log.loc[i].name
  range_value = log.loc[i].range
  ax3.text(30, y, f'{loc}: {range_value}', fontsize=60, ha='left', style='italic')
  y-=1.8

# WWA Legend setup

wwa_ax.set_extent([-109.05, -102.05, 37, 41])
wwa_ax.add_feature(cfeature.STATES)
plt.rcParams['font.family'] = treb.get_name()
counties.plot(ax=wwa_ax, color='none', ec='gray', alpha=0.75)
interstates.plot(ax=wwa_ax, ec='r', linewidth=1.5)
cdot.plot(ax=wwa_ax, aspect=1, fc='none', ec='k', linewidth=2, zorder=12)
for i in range(len(co)):
    wwa_ax.add_geometries(co.loc[i].geometry, fc=co.loc[i]['Warning Color'], crs=ccrs.PlateCarree())
prod_types = co['PROD_TYPE'].unique()
colors = [co[co['PROD_TYPE'] == pt]['Warning Color'].iloc[0] for pt in prod_types]

if len(co) == 0:
    # If `co` is empty, create a legend saying "None"
    legend_patches = [mpatches.Patch(color='gray', label='None')]  
else:
    # If `co` is not empty, proceed with generating the legend as usual
    prod_types = co['PROD_TYPE'].unique()
    colors = [co[co['PROD_TYPE'] == pt]['Warning Color'].iloc[0] for pt in prod_types]
    legend_patches = [mpatches.Patch(color=color, label=wrap_label(pt, width=10)) for pt, color in zip(prod_types, colors)]

ax1.set_position(ax1pos)
ax3.set_position(ax3pos)

legend = legend_ax.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.65, 0.5),
                          fontsize=50, title=wrap_label("NWS Watches, Warnings, and Advisories", width=16), title_fontsize=60, frameon=False)

legend_ax.axis('off')
legend_ax.set_position(legend_axpos)
cax.set_position(caxpos)
wwa_ax.set_position(wwa_axpos)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
# Graphic Design adds

fig.patches.append(mpatches.Rectangle((0, 0.95), 1, 0.13, transform=fig.transFigure, facecolor='#F2F2F2', zorder=-1))
fig.patches.append(mpatches.Rectangle((0, 0.940), 1, 0.015, transform=fig.transFigure, facecolor='#EF7521', zorder=-1))
fig.patches.append(mpatches.Rectangle((0.025, 0.0075), 0.01, 0.385, transform=fig.transFigure, facecolor='#001970', zorder=15))
fig.patches.append(mpatches.Rectangle((0.305, 0.0075), 0.01, 0.385, transform=fig.transFigure, facecolor='#001970', zorder=15))

# Final touches and exporting the figure
# If this part uses too much RAM, add dpi=50 to the end of the last line within the )
# The default value is 100, so just making a smaller export figure will use less memory

fig.suptitle(' ', x=0, y=1.07, fontsize=240, ha='left')
fig.text(0.025, 1.004, f'Winter Weather Report: {todayst}', fontsize=200, weight='bold', style='italic')
fig.text(0.04, 0.967, f'Valid as of {ts}', fontsize=80, weight='bold', style='italic')
fig.figimage(cdotlogo, 5500, 4250, zorder=20)
fig.figimage(nws, 8500, 4325, zorder=20)
file_path = os.path.join(base_dir, f'{todaystr}_MWR.png')
plt.savefig(file_path, bbox_inches='tight')

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Gmail SMTP settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Sender's email and App Password (for Gmail 2FA)
SENDER_EMAIL = "beckerman1101@gmail.com"
SENDER_PASSWORD = os.getenv('GMAIL_PW')

# Recipient email
RECIPIENT_EMAILS = ["brendan.eckerman@state.co.us", "michael.chapman@state.co.us", "nicholas.barlow@state.co.us","bob.fifer@state.co.us","shawn.smith@state.co.us","james.fox@state.co.us","phillip.embry@state.co.us"]

# Email Subject & Body
SUBJECT = f"Morning Weather Report - {todayst}"
BODY = "Attached is today's Morning Weather Report. If you have questions or the report appears to have had issues in the process of its creation, please reach out to Brendan at brendan.eckerman@state.co.us\n\nPlease note that accumulated snowfall data is preliminary and has not been refined for quality control. It is also derived from radar data, which is susceptible to errors along terrain features."

# Path to the PNG file
ATTACHMENT_PATH = file_path # Update with your PNG file path

def send_email():
    # Create the MIME email object
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECIPIENT_EMAILS)
    msg['Subject'] = SUBJECT

    # Attach the email body
    msg.attach(MIMEText(BODY, 'plain'))

    # Attach the PNG file
    try:
        with open(ATTACHMENT_PATH, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(ATTACHMENT_PATH)}')
            msg.attach(part)
    except Exception as e:
        print(f"Error attaching file: {e}")
        return

    # Send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Encrypts the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Log in using email and App Password
            text = msg.as_string()
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, text)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")
fcst_trigger = log['fcst'].values.max()
accum_trigger = log['accum'].values.max()
# Main function to call the send_email function
if __name__ == "__main__":
    if fcst_trigger >= 4 or accum_trigger >= 2 or len(co) != 0:
        send_email()
    else:
        print('Snow thresholds not met for distribution')
