from armor_tools import plot as plotter
from armor_tools import analysis 
from pathlib import Path
import pyart
import numpy as np
import pandas as pd


ROOT = Path('/nas/rhome/eebbert/armor/cfrad')

f_rhi = ROOT / 'ARMR20250820183028.nc'

f_snd = Path('/nas/rhome/eebbert/2025082018-72800.csv')

f_processed = Path('/nas/rhome/eebbert/ARMOR_20250820_183028_RHI_ZH_v1.nc')

import xarray as xr

ds = xr.open_dataset(f_processed)

print(ds)

radar = pyart.io.read(f_rhi)
print(radar.fields.keys())

# QC 

# Values for clutter filtering
SNR = 5 # SNR threshold
RHO = 0.6 #rho hv threshold
fields_to_filter = ['REF', 'VEL', 'ZDR', 'RHO', 'PHI', 'SW']

# pointing angle offset
el_offset = 0.3

# correcting pointing angle
radar = analysis.correct_elevation_pointing_angle(radar, offset=el_offset)

# clutter filtering
for field in fields_to_filter:
    if field in radar.fields:
        radar = analysis.noise_filter(radar, field, SNR=SNR, rho=RHO)
    else:
        print(f'{field} not found in {f.name}')

# velocity dealiasing using Py-ART texture filter
if 'FVEL' in radar.fields:
    radar = analysis.dealias_velocity(radar, vel_field='FVEL')
else:
    print(f'FVEL not found in {f.name}, skipping dealiasing.')

# plotting
fields = ['FREF', 'FVEL', 'FZDR', 'FPHI', 'FRHO']

plotter.plot_rhi(radar=radar, fields=fields)

