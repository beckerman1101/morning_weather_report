CDOT MORNING WEATHER REPORT

This repository is set up to compile Colorado weather information from various NWS sources, then generate a graphic for daily distribution within the internal CDOT network for operational purposes. 

The images and shapefiles for the graphic are housed in their respective folders, and everything for generating the report is contained wihin morning_weather_report.py. Font files are necessary for CDOTs approved font, Trebuchet MS. 
The python script will run daily at 10:45 UTC, or 4:45 AM locally during MDT. It begins by importing all the necessary python packages, detailed in the trequirements.txt file. 
It then establishes strings for today, yesterday, and other date variables for filenaming and other necessary purposes.

Next is establishing environment variables and other necessary things within the repository

Font managing is next, before loading all of the images and shapefiles and processing them for use. 

Then colorbars. The snow colorbar is the same as the one utilized by the NWS, with sharp color demarcations at 6in and 18in
The WWA colorbar is currently only initialized for common NWS watch, warning , and advirosy products in Colorado. More can be added later if deemed necessary, but for now, they serve their purpose. 

Next up is querying and establishing readable datasets for the AFD, WWA products, NDFD snowfall data, and NOHRSC Gridded snowfall analysis, which uses radar, HRRR initializations, and climatological SLR data to produce accumulated snowfall. 
The snowfall data is then processed for 10 high-traffic locations in and around the main urban corridors of Colorado, which are used later for distribution purposes.
Snowfall data also gets interpolated for smoother plotting. 

Initializing and creating the figure is the longest chunk of the python script, between lines 345 and 506. This should always be the smoothest-running portion of the script. 
Finally, functions are defined to send the generated .png file to a list of the necessary recipients within CDOTs network, mainly the weather team and managerial positions in the Division of Maintenance and Operations.

The email only gets sent if there are more than 4 inches of snowfall at any of the selected points from earlier in the next 48 hours, or 2 inches at those locations from midnight yesterday to midnight this morning. 
