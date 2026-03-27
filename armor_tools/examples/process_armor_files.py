'''
This script is for quality controlling ARMOR data. Some features of this script are: filtering scan type for a given data range, correcting elevation pointing angle values, and more!

This script expects CFRadial data zipped in .xz format organized in folders based on date (Exactly how the data are formatted on MATRIX at the directory /armor/cfrad). 

'''

# imports
from armor_tools import analysis
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import pyart


# USER DEFINED VARIABLES
# root directory is where your data folders, organized by date are located
ROOT = Path('/nas/rhome/eebbert/armor/cfrad/')

# (year, month, date, hour, minute, second)
start_datetime = datetime(2025, 5, 20, 0, 0, 0)
end_datetime   = datetime(2025, 5, 21, 0, 0, 0)

# output directory
start_str = start_datetime.strftime('%Y%m%d')
end_str = end_datetime.strftime('%Y%m%d')
date_str = start_str if start_str == end_str else f"{start_str}_{end_str}"
output = Path(f'/wopr1/timeslice/armor_BNF')
output.mkdir(parents=True, exist_ok=True)


# elevation angle offset for RHI scan corrections
el_offset = 0.3

# Values for clutter filtering
SNR = 5 # SNR threshold
RHO = 0.6 #rho hv threshold
fields_to_filter = ['REF', 'VEL', 'ZDR', 'RHO', 'PHI', 'SW']


# QUALITY CONTROL SCRIPT DON'T EDIT ANYTHING BELOW THIS

# finding files based on user defeined date and time
print(f'User defined folder: {ROOT}')
print(f'Searching for files between {start_datetime} and {end_datetime} ...')
files = analysis.find_files_in_timerange(ROOT, start_datetime, end_datetime)
print(f'Found {len(files)} matching files.')

# filtering files based on VCP number
print('Filtering by scan type...')
rhi_files = analysis.filter_files_vcp(files, vcp_min=0, vcp_max=100)
print(f'RHI Files: {len(rhi_files)}')
ppi_files = analysis.filter_files_vcp(files, vcp_min=200, vcp_max=300)
print(f'PPI Files: {len(ppi_files)}')
ppi_sector_files = analysis.filter_files_vcp(files, vcp_min=100, vcp_max=200)
print(f'PPI Sector Files: {len(ppi_sector_files)}')
# other files outside of VCP filter
used_files = set(rhi_files) | set(ppi_files) | set(ppi_sector_files)
other_files = [f for f in files if f not in used_files]
print(f'Other/Unclassified Files: {len(other_files)}')


# QC Steps:
# Pointing Angle Correction
# Clutter Filter
# Velocity Dealias
# Save as new .nc file

# QC-ing RHI Files
for f in tqdm(rhi_files, desc='QC-ing RHI Files', unit='file'):
    f_nc = None
    try:
        # decompressing file
        f_nc = analysis.decompress_xz(f)

        # reading in file
        radar = pyart.io.read(f_nc)

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

        output_dir_rhi = output / 'rhi'
        output_dir_rhi.mkdir(parents=True, exist_ok=True)
        # saving corrected radar object to .nc file
        analysis.radar_to_nc(radar=radar, original_file=f, output_dir=output_dir_rhi)

    except Exception as e:
        print(f'Error processing {f.name}: {e}')

    finally:
        # removing temp f_nc file
        if f_nc is not None:
            analysis.remove_nc(f_nc)

# QC Steps:
# Pointing Angle Correction
# Clutter Filter
# Velocity Dealias
# Save as new .nc file
# QC-ing PPI Files

for f in tqdm(ppi_files, desc='QC-ing PPI Files', unit='file'):
    # quality control functions
    f_nc = None
    try:
        # decompressing file
        f_nc = analysis.decompress_xz(f)

        # reading in file
        radar = pyart.io.read(f_nc)

        # correcting pointing angle
        radar = analysis.correct_azimuth_pointing_angle_ppi_dynamic(radar)

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

        output_dir_ppi = output / 'ppi'
        output_dir_ppi.mkdir(parents=True, exist_ok=True)
        # saving corrected radar object to .nc file
        analysis.radar_to_nc(radar=radar, original_file=f, output_dir=output_dir_ppi)

    except Exception as e:
        print(f'Error processing {f.name}: {e}')

    finally:
        # removing temp f_nc file
        if f_nc is not None:
            analysis.remove_nc(f_nc)

# QC Steps:
# Pointing Angle Correction
# Clutter Filter
# Velocity Dealias
# Save as new .nc file

# QC-ing PPI Sector Files
for f in tqdm(ppi_files, desc='QC-ing PPI Sector Files', unit='file'):
    # quality control functions
    f_nc = None
    try:
        # decompressing file
        f_nc = analysis.decompress_xz(f)

        # reading in file
        radar = pyart.io.read(f_nc)

        # correcting pointing angle
        radar = analysis.correct_azimuth_pointing_angle_sector(radar, verbose=False)

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

        output_dir_sec = output / 'sec'
        output_dir_sec.mkdir(parents=True, exist_ok=True)
        # saving corrected radar object to .nc file
        analysis.radar_to_nc(radar=radar, original_file=f, output_dir=output_dir_sec)

    except Exception as e:
        print(f'Error processing {f.name}: {e}')

    finally:
        # removing temp f_nc file
        if f_nc is not None:
            analysis.remove_nc(f_nc)
