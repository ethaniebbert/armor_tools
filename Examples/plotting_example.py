import armor_tools.analysis as an
import armor_tools.plot as aplt
from pathlib import Path
import pyart

# Defining Fields to be plotted, change to what you want, options are:
# REF, VEL, SW, RHO, ZDR
fields = ['REF', 'VEL', 'ZDR']

# update root to the directory path containing your data
root = Path('/Users/eebbert/Downloads/example_data')



# path structure for example files and plots, will output the plot folders inside the root directory
# you shouldn't need to change these unless you want the plots saved to a different directory than the directory containing the data folder
DATA = root / 'plotting_data'
PPI_PLOTS = root / 'ppi_plots'
RHI_PLOTS = root / 'rhi_plots'

# Create plotting directories if they don't exist
PPI_PLOTS.mkdir(parents=True, exist_ok=True)
RHI_PLOTS.mkdir(parents=True, exist_ok=True)

# === RHI PLotting Example ===
'''
Workflow:
Filter the data folder for only RHI scans
Loop through the list of files:
- decompress the .xz file into a temp .nc file
- read in the temp file as a pyart radar object
- plot the rhi scan using the radar object
- remove the temp nc file
'''

# Filtering cfrad files for RHI scans
filtered_rhi_data = an.filter_folder_vcp(DATA, 0, 99)

# plotting RHI scans
for f in filtered_rhi_data:
    f_nc = an.decompress_xz(f)
    radar = pyart.io.read(f_nc)
    aplt.plot_rhi(radar, fields=fields, save_path=RHI_PLOTS, grids=True, xmax=60, ymax=12)
    an.remove_nc(f_nc)

# === PPI PLotting Example ===
'''
Workflow:
Filter the data folder for only PPI scans
Loop through the list of files:
- decompress the .xz file into a temp .nc file
- read in the temp file as a pyart radar object
- plot the rhi scan using the radar object
- remove the temp nc file
'''

# We're only plotting the first sweep (in this case the lowest)
sweeps = [0]

# Filtering cfrad files for PPI scans
filtered_ppi_data = an.filter_folder_vcp(DATA, 200, 299)

# plotting RHI scans
for f in filtered_ppi_data:
    f_nc = an.decompress_xz(f)
    radar = pyart.io.read(f_nc)
    aplt.plot_ppi(radar, sweeps=sweeps, fields=fields, save_path=PPI_PLOTS, grids=True, xmin=-120, ymin=-120, xmax=120,
                  ymax=120)
    an.remove_nc(f_nc)