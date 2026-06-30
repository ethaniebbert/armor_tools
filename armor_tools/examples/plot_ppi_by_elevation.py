'''
Generate two-panel PPI plots (FREF and FVEL) for all sweeps in a folder of
ARMOR PPI files. Plots are organized into subdirectories by elevation angle:

    /nas/rhome/eebbert/armor/plots/ppi/YYYYMMDD/el_X.X/ARMR_PPI_*.png

Handles both raw .nc.xz (compressed) and pre-processed .nc files. Falls back
to REF/VEL if filtered fields are not present.
'''

from armor_tools import analysis
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import pyart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# USER DEFINED VARIABLES

# Folder containing PPI files to plot (.nc or .nc.xz)
INPUT_FOLDER = Path('/nas/rhome/eebbert/armor/filtered/20260622_20260623/ppi/')

# Base output directory (YYYYMMDD subfolder is created automatically)
PLOT_ROOT = Path('/nas/rhome/eebbert/armor/plots/ppi')

rng = 50 # km

# Plot extent in km
XMIN, XMAX = -rng, rng
YMIN, YMAX = -rng, rng


# SCRIPT — DON'T EDIT BELOW THIS LINE

FIELD_PARAMS = {
    'FREF': {'vmin': -10, 'vmax': 70,  'cmap': 'HomeyerRainbow', 'label': 'Reflectivity (dBZ)'},
    'REF':  {'vmin': -10, 'vmax': 70,  'cmap': 'HomeyerRainbow', 'label': 'Reflectivity (dBZ)'},
    'FVEL': {'vmin': -16, 'vmax': 16,  'cmap': 'PuOr_r',         'label': 'Radial Velocity (m/s)'},
    'VEL':  {'vmin': -16, 'vmax': 16,  'cmap': 'PuOr_r',         'label': 'Radial Velocity (m/s)'},
}


def _resolve_field(radar, preferred, fallback):
    """Return preferred field name if present, else fallback, else None."""
    if preferred in radar.fields:
        return preferred
    if fallback in radar.fields:
        print(f'  {preferred} not found, falling back to {fallback}')
        return fallback
    print(f'  Neither {preferred} nor {fallback} found, skipping.')
    return None


def _sweep_datetime(radar, sweep):
    units_str = radar.time['units']
    base_time_str = units_str.split('since')[-1].strip().replace('Z', '')
    base_time = datetime.fromisoformat(base_time_str)
    ray0 = int(radar.sweep_start_ray_index['data'][sweep])
    return base_time + timedelta(seconds=round(float(radar.time['data'][ray0])))


def plot_sweep(radar, sweep, ref_field, vel_field, out_dir):
    display = pyart.graph.RadarDisplay(radar)
    ray0 = int(radar.sweep_start_ray_index['data'][sweep])
    elevation = float(radar.elevation['data'][ray0])
    sweep_dt = _sweep_datetime(radar, sweep)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, field in zip(axes, [ref_field, vel_field]):
        p = FIELD_PARAMS[field]
        display.plot_ppi(
            field, sweep=sweep, ax=ax,
            vmin=p['vmin'], vmax=p['vmax'],
            cmap=p['cmap'],
            colorbar_label=p['label'],
            title=f"ARMOR PPI | {p['label']}\nEl: {elevation:.1f}°  |  {sweep_dt.isoformat()}Z"
        )
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel('Distance from Radar (E/W) (km)')
        ax.set_ylabel('Distance from Radar (N/S) (km)')

    plt.tight_layout()

    safe_time = sweep_dt.strftime('%Y%m%d%H%M%S')
    out_file = out_dir / f'ARMR_PPI_{safe_time}_el{elevation:.1f}deg.png'
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


# Collect files: prefer .nc.xz; include .nc only if no compressed twin exists
xz_files = sorted(INPUT_FOLDER.glob('ARMR*.nc.xz'))
xz_nc_names = {f.stem for f in xz_files}   # stems look like 'ARMR*.nc'
nc_only = [f for f in sorted(INPUT_FOLDER.glob('ARMR*.nc')) if f.name not in xz_nc_names]
files = xz_files + nc_only

print(f'Found {len(files)} PPI file(s) in {INPUT_FOLDER}')

for f in tqdm(files, desc='Plotting PPI files', unit='file'):
    f_nc = None
    try:
        if f.suffix == '.xz':
            f_nc = analysis.decompress_xz(f)
            radar_path = f_nc
        else:
            radar_path = f

        radar = pyart.io.read(str(radar_path))

        # Date for top-level output subfolder
        units_str = radar.time['units']
        base_time_str = units_str.split('since')[-1].strip().replace('Z', '')
        date_str = datetime.fromisoformat(base_time_str).strftime('%Y%m%d')

        ref_field = _resolve_field(radar, 'FREF', 'REF')
        vel_field = _resolve_field(radar, 'FVEL', 'VEL')

        if ref_field is None or vel_field is None:
            print(f'  Skipping {f.name}: required fields missing.')
            continue

        for sweep in range(radar.nsweeps):
            ray0 = int(radar.sweep_start_ray_index['data'][sweep])
            elevation = float(radar.elevation['data'][ray0])
            el_folder = f'el_{elevation:.1f}'

            out_dir = PLOT_ROOT / date_str / el_folder
            out_dir.mkdir(parents=True, exist_ok=True)

            plot_sweep(radar, sweep, ref_field, vel_field, out_dir)

    except Exception as e:
        print(f'Error processing {f.name}: {e}')

    finally:
        if f_nc is not None:
            analysis.remove_nc(f_nc)

print('Done.')
