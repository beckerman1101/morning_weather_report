#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Python packages
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
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import cfgrib
import tqdm
from matplotlib.colors import ListedColormap
import requests
from zoneinfo import ZoneInfo
import zipfile
import os


# In[2]:
GOOGLE_EMAIL = os.getenv("GOOGLE_EMAIL")
GOOGLE_PASSWORD = os.getenv("GOOGLE_PASSWORD")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS")

#Map Variable Setup

font1 = {'fontname':'Trebuchet MS', 'weight':'bold'}
font2 = {'fontname':'Trebuchet MS'}
#font3 = {'fontname':'Franklin Gothic Medium'}

#Logos
cdotlogo = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/image (1).png')
us160 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/US_160.png')
us550 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/US_550.png')
us50 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/US_50.png')
us40 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/US_40.png')
us285 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/US_285.png')
i25 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/I-25.png')
i70 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/I-70.png')
i76 = mpimg.imread('C:/Users/eckermanbe/Desktop/Python/I-76.png')

#Colorado bounds and padding
co_bounds = [-109.05,-102.05,37,41]
x = .15

#Placefiles for counties, interstates, state highways, and manipulating the regions shapefile to plot properly
counties = gpd.read_file('C:/Users/eckermanbe/Desktop/Python/tl_2019_08_county.shx')
cdot = gpd.read_file('C:/Users/eckermanbe/Desktop/Python/MaintenanceSections.shp').set_crs(epsg=26913)
cdot = cdot.to_crs(epsg=32662)
cdot_bounds = cdot.total_bounds
x_scale = (co_bounds[1]-co_bounds[0])/(cdot_bounds[2]-cdot_bounds[0])
y_scale = (co_bounds[3]-co_bounds[2])/(cdot_bounds[3]-cdot_bounds[1])
cdot = cdot.scale(x_scale, y_scale, origin=(0,0))
cdot_bounds2 = cdot.total_bounds
lx = cdot_bounds2[0]
ly = cdot_bounds2[1]
cdot = cdot.translate(xoff=co_bounds[0]-lx, yoff=co_bounds[2]-ly)
sh = gpd.read_file('C:/Users/eckermanbe/Desktop/Python/tl_2021_08_prisecroads.shp')
interstates = gpd.read_file('C:/Users/eckermanbe/Desktop/Python/us_interstate_highways.shp')


# In[3]:


#Colorbar Setup
#Snowfall colorbar
colors_snow = [
    '#FFFFFF', '#E7E7E7', '#CBE0ED', '#89BFDF', '#5A9CCB', '#3675B2', '#3650AB',
    '#FFFFAD', '#FFD130', '#FFA030', '#E34130', '#B23030', '#883030', '#D7D7FF', '#B3A4E1',
    '#9775B8', '#7B458F', '#5A305B']
cmaps = mcolors.ListedColormap(colors_snow)
boundariess = [0,0.1,1,2,3,4,5,6,8,10,12,14,16,18,24,30,36,42]
norms = mcolors.BoundaryNorm(boundariess, cmaps.N)
snow_labels = ['',0.1,1,2,3,4,5,6,8,10,12,14,16,18,24,30,36,'']


#Black colorbar
black_cmap = ListedColormap([(0, 0, 0, 1)])


# In[4]:


### ---- PLOTTING CELL ---- ###

## Initialize today's date, format in order to get today's snow file, then open the file

today = datetime.today()
yesterday = today - timedelta(days=1)
todayst = today.strftime('%a %m/%d')
yesterdayst = yesterday.strftime('%a %m/%d')
todaystr = today.strftime('%Y%m%d')

url = f"https://www.nohrsc.noaa.gov/snowfall_v2/data/202502/sfav2_CONUS_24h_{todaystr}12.nc"
output_dir = 'C:/Users/eckermanbe/Desktop/Python/Daily Files'
output_file = os.path.join(output_dir, f"{todaystr}_file.nc")
response = requests.get(url, stream=True)
with open(output_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

# Open the dataset
file_path = os.path.join(output_dir, f'{todaystr}_file.nc')
file = xr.open_dataset(file_path)


## Initialize the figure

fig, ax = plt.subplots(figsize=(88,67), constrained_layout=True, subplot_kw=dict(projection = ccrs.PlateCarree()))
ax.set_extent([co_bounds[0]-x, co_bounds[1]+x, co_bounds[2]-x, co_bounds[3]+x])
plt.rcParams['font.family'] = 'Trebuchet MS'
fig.add_axes([0, -0.01, 1, 0.11])  # [left, bottom, width, height]
plt.axis('off')


## Data processing

df = 39.3701*file
df_co = df.where(df.lat>=co_bounds[2], drop=True).where(df.lat<=co_bounds[3], drop=True).where(df.lon>=co_bounds[0], drop=True).where(df.lon<=co_bounds[1], drop=True)
latlen = len(df_co.lat)
lonlen = len(df_co.lon)
inter_factor = 8
target_lat = np.linspace(co_bounds[2], co_bounds[3], latlen*inter_factor)  # Double the resolution of latitudes
target_lon = np.linspace(co_bounds[0], co_bounds[1], lonlen*inter_factor)  # Double the resolution of longitudes
df = df_co['Data'].interp(lat=target_lat, lon=target_lon, method='linear')    #Add borders, highways to the map


## Plot highways, counties, CDOT regions

ax.add_feature(cfeature.STATES, ec='k', linewidth=2.5)
counties.plot(ax=ax, color='none', ec='gray', alpha=0.75)
sh.plot(ax=ax, ec='r', linewidth=4, alpha=0.7)
interstates.plot(ax=ax, ec='r', linewidth=8)
cdot.plot(ax=ax, aspect=1, fc='none', ec='k', linewidth=12, zorder=12)


## Plot the snow data and its colorbar

snow_map = df.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(), cmap=cmaps, norm=norms, add_colorbar=False)
cbar = fig.colorbar(snow_map, ax=ax, orientation='horizontal', ticks=boundariess, location='bottom')
cbar.ax.set_xticklabels(snow_labels)
cbar.ax.tick_params(labelsize=90, pad=20)


##Process and plot point data: changes can be made to locations, always filters data below 0.5"

table = pd.read_csv('C:/Users/eckermanbe/Desktop/Python/map_locations.csv')
lats = []
lons = []
vals = []
for i in tqdm.tqdm(range(len(table))):
    lat = []
    lon = []
    a = table.loc[i].lat
    b = table.loc[i].lon
    lat.append(a)
    lon.append(b)
    val = df.sel(lat=a, method='nearest').sel(lon=b, method='nearest').values.max()
    lats.append(lat[0])
    lons.append(lon[0])
    vals.append(val)
log = pd.DataFrame({'lat':lats, 'lon':lons, 'snow':vals})
point_snow = log[log['snow'] >= 0.5].copy()
point_snow['snow'] = point_snow['snow'].round().astype(int)
ax.scatter(point_snow['lon'], point_snow['lat'], c=point_snow['snow'], cmap=black_cmap, s=15000, zorder=25, edgecolors='black', transform=ccrs.PlateCarree())
for i, row in point_snow.iterrows():
    ax.text(row['lon'], row['lat'], f'{int(row["snow"])}"', fontsize=75, ha='center', va='center', zorder=30, color='white', transform=ccrs.PlateCarree(), **font1)


## Plotting Highway logos

ax.imshow(i25, extent=[-104.90, -104.75, 37.75, 37.9], zorder=15)
ax.imshow(i25, extent=[-105.06, -104.91, 40.35, 40.5], zorder=15)
ax.imshow(i70, extent=[-107.20, -107.05, 39.55, 39.7], zorder=15)
ax.imshow(i70, extent=[-103.75, -103.6, 39.18, 39.33], zorder=15)
ax.imshow(i76, extent=[-103.6, -103.45, 40.2, 40.35], zorder=15)
ax.imshow(us40, extent=[-106.23, -106.13, 40, 40.1], zorder=15)
ax.imshow(us40, extent=[-108, -107.9, 40.46, 40.56], zorder=15)
ax.imshow(us50, extent=[-107.3, -107.2, 38.42, 38.52], zorder=15)
ax.imshow(us50, extent=[-104.09, -103.99, 38.08, 38.18], zorder=15)
ax.imshow(us550, extent=[-107.89, -107.76, 37.45, 37.55], zorder=15)
ax.imshow(us285, extent=[-105.63, -105.50, 39.4, 39.5], zorder=15)
ax.imshow(us160, extent=[-106.78, -106.65, 37.55, 37.65], zorder=15)
ax.imshow(us160, extent=[-103.14, -103.01, 37.22, 37.32], zorder=15)


## Adding titles and CDOT Logo

fig.suptitle(' ', x=0.005, y=.99, ha='left', fontsize=275)
fig.text(0.014, 0.945, 'NWS Interpolated Snowfall Accumulation', fontsize=200, **font1)
fig.text(0.028, 0.105, 'https://https://www.nohrsc.noaa.gov/snowfall_v2/', fontsize=94)
fig.text(0.875, 0.105, 'NWS Gridded Snowfall Analysis', fontsize=94, ha='center')
fig.text(0.028, 0.92, f'Valid: 5am {yesterdayst} - 5am {todayst}', fontsize=85)
fig.figimage(cdotlogo, 5800, 6080, zorder=35)
path = f'C:/Users/eckermanbe/Desktop/Python/24houraccum_{todaystr}.png'
plt.savefig(path, bbox_inches='tight', pad_inches=0.01)


# In[5]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path):
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)  # Convert list to comma-separated string
    msg['Subject'] = subject

    # Attach the body of the email
    msg.attach(MIMEText(body, 'plain'))

    # Open the attachment file in binary mode
    with open(attachment_file_path, 'rb') as attachment_file:
        # Attach the file to the email
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_file.read())
        encoders.encode_base64(part)  # Encode the attachment
        part.add_header('Content-Disposition', f'attachment; filename={attachment_file_path.split("/")[-1]}')
        msg.attach(part)

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, f"{GOOLE_PASSWORD}")  # Use an app password if 2FA is enabled
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            print(f'Email sent successfully to: {", ".join(receiver_emails)} with attachment: {attachment_file_path}')
    except Exception as e:
        print(f"Failed to send email: {e}")

# Example usage
sender_email = f"{GOOGLE_EMAIL}"
receiver_emails = [f"{EMAIL_RECIPIENTS}"]  # Add multiple emails in a list
subject = "24-Hour Snowfall Report"
body = "This email is automated. Attached is the report for snowfall statewide in the past 24 hours. Data is preliminary and has not been refined for quality control."
attachment_file_path = path  # Make sure the path is correct

send_email_with_attachment(sender_email, receiver_emails, subject, body, attachment_file_path)





# In[ ]:




