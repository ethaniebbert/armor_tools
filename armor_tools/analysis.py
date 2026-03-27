'''
The purpose of this Python file is to serve as a place for all data analysis functions to live.
'''
import pyart
import lzma
import gzip
from pathlib import Path
import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List

def decompress_xz(input_file, output_file=None):
    """
    Decompress a .xz-compressed CfRadial NetCDF file to disk.

    Parameters
    ----------
    input_file : str or Path
        Path to the .xz file.
    output_file : str or Path, optional
        Path where the decompressed .nc file will be written.
        If None, uses the same name as input_file with `.nc` extension.

    Returns
    -------
    Path
        Path to the decompressed .nc file.
    """
    input_file = Path(input_file)

    if output_file is None:
        # strip `.xz` and replace with `.nc`
        output_file = input_file.with_suffix("")
        if output_file.suffix != ".nc":
            output_file = output_file.with_suffix(".nc")

    output_file = Path(output_file)

    with lzma.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(f_in.read())

    return output_file

def decompress_gz(input_file, output_file=None):
    """
    Decompress a .gz-compressed CfRadial NetCDF file to disk.

    Parameters
    ----------
    input_file : str or Path
        Path to the .gz file.
    output_file : str or Path, optional
        Path where the decompressed .nc file will be written.
        If None, uses the same name as input_file with `.nc` extension.

    Returns
    -------
    Path
        Path to the decompressed .nc file.
    """
    input_file = Path(input_file)

    if output_file is None:
        # strip `.gz` and replace with `.nc`
        output_file = input_file.with_suffix("")
        if output_file.suffix != ".nc":
            output_file = output_file.with_suffix(".nc")

    output_file = Path(output_file)

    with gzip.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(f_in.read())

    return output_file


def remove_nc(file_path):
    """
    Remove a decompressed .nc file from disk.

    Parameters
    ----------
    file_path : str or Path
        Path to the .nc file to remove.

    Returns
    -------
    bool
        True if the file was removed, False if it didn't exist.
    """
    file_path = Path(file_path)

    if file_path.exists() and file_path.suffix == ".nc":
        os.remove(file_path)
        return True
    else:
        return False

def list_fields(nc_filepath):
    """
        Lists fields contained in a netCDF ARMOR file

        Parameters
        ----------
        nc_filepath : str or Path
            Path to the .nc file to read.

        Returns
        -------
        list
            List contaning field names
        """
    ds = xr.open_dataset(nc_filepath)
    vars = list(ds.keys())
    return vars

def L2_to_CFRad(L2_filename, save_path):
    """
    Convert a NEXRAD Level II radar file to CfRadial format.

    This function reads a NEXRAD Level II archive file using Py-ART
    and writes it out as a CfRadial-compliant NetCDF file. The output
    filename matches the input name, but with a `.nc` extension, and is
    saved to the specified directory.

    Parameters
    ----------
    L2_filename : str or Path
        Path to the NEXRAD Level II file to convert.

    save_path : str or Path
        Directory where the converted CfRadial file will be written.

    Returns
    -------
    str
        Full path to the output CfRadial `.nc` file.
    """
    #strips the name of the file
    base = os.path.splitext(os.path.basename(L2_filename))[0]
    #adds the name to the save_path
    out_path = os.path.join(save_path, f"{base}.nc")
    #reads in the L2 file with pyart and converts it
    radar = pyart.io.read_nexrad_archive(L2_filename)
    pyart.io.write_cfradial(out_path, radar)

    return out_path

def cal_beam_height(rng, elev_angle):
    """
    Function to calculate the beam height of a radar

    Parameters
    ----------
    rng : int or float
        range to a radar gate in meters

    elev_angle : int or float
        elevation angle for a radar gate in degrees

    Returns
    -------
    float
        height of range gate in meters
    """

    # Constants
    A_E = 6371000 # Avg. Earth Radius (meters)
    K = 4/3 # Refraction Term, assumes standard refraction of 4/3 ae

    # Converting elev_angle to radians
    rad_angle = elev_angle * (np.pi/180)

    # Calculating Beam Height Above Radar Level
    z_arl = np.sqrt(rng**2 + (K*A_E)**2 + 2*rng*K*A_E*np.sin(rad_angle)) - K*A_E

    return z_arl

def cal_elev_angle(rng, z_arl):
    """
    Function to calculate the elevation angle of a radar

    Parameters
    ----------
    rng : int or float
        range to a radar gate in meters

    z_arl : int or float
        height above radar level for a radar gate in meters

    Returns
    -------
    float
        elevation angle for range gate in degrees
    """

    # Constants
    A_E = 6371000  # Avg. Earth Radius (meters)
    K = 4 / 3  # Refraction Term, assumes standard refraction of 4/3 ae

    # Solving for elevation angle in radians
    rad_angle = np.arcsin(((z_arl + K * A_E) ** 2 - (K * A_E) ** 2 - rng ** 2) / (2 * rng * K * A_E))

    # Converting elevation angle to degrees
    elev_angle = rad_angle * (180/np.pi)

    return elev_angle


def filter_folder_vcp(file_folder, vcp_min, vcp_max):
    """
    Return a list of .xz CfRadial files whose VCP value falls within
    the interval [vcp_min, vcp_max). No files are deleted.

    Parameters
    ----------
    file_folder : str or Path
        Path to the folder containing `.xz` files to filter.

    vcp_min : int
        Minimum VCP value to retain (inclusive).

    vcp_max : int
        Maximum VCP value to retain (exclusive).

    Returns
    -------
    list of Path
        Files whose VCP values fall within the specified range.
    """
    file_path = Path(file_folder)
    files = sorted(file_path.glob("*.xz"))
    kept_files = []

    for f in files:
        # decompress to temp .nc
        f_nc = decompress_xz(f)

        # read VCP
        ds = xr.open_dataset(f_nc)
        try:
            vcp = int(ds.vcp_pattern)
        finally:
            ds.close()

        # keep file if within range
        if vcp_min <= vcp < vcp_max:
            kept_files.append(f)

        # remove the temporary .nc
        remove_nc(f_nc)

    return kept_files


def correct_elevation_pointing_angle(radar, offset=0.30):
    """
    Apply a pointing-angle correction to radar elevation data.
    Modifies the radar object in-place and returns it.

    Parameters
    ----------
    radar: Py-ART radar object
    offset: float
        elevation correction in degrees to apply to elevation values. Default is .30 degrees, which is valid for around 3 degrees/s of elevation scan speed.
    
    Returns
    -------
    radar
        Py-ART radar object with elevation corrections applied to all sweeps
    """

    for sweep in range(radar.nsweeps):

        start = radar.sweep_start_ray_index['data'][sweep]
        end = radar.sweep_end_ray_index['data'][sweep] + 1

        elev = radar.elevation['data'][start:end]

        # Determine sweep direction from elevation trend
        if elev[-1] > elev[0]:      # bottom-to-top
            radar.elevation['data'][start:end] += offset
        else:                        # top-to-bottom
            radar.elevation['data'][start:end] -= offset

    return radar

def correct_azimuth_pointing_angle_ppi_dynamic(radar):
    """
    Apply a scan-speed-dependent azimuth correction per sweep.

    Sweep direction is determined from the azimuth trend within each sweep,
    similar to the elevation correction function.

    Uses empirical relationship from ARMOR pointing study:
        correction ≈ 0.133 * (scan_speed - 10)

    Returns
    -------
    radar
        Py-ART radar object with azimuth corrections applied per sweep
    """

    for sweep in range(radar.nsweeps):

        start = radar.sweep_start_ray_index['data'][sweep]
        end = radar.sweep_end_ray_index['data'][sweep] + 1

        az = np.asarray(radar.azimuth['data'][start:end])
        time_vals = np.asarray(radar.time['data'][start:end])

        if len(az) < 2:
            continue

        # unwrap azimuth so 359 -> 0 behaves correctly
        az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(az)))

        # determine sweep direction from azimuth trend
        if az_unwrapped[-1] > az_unwrapped[0]:
            direction = 'cw'
        else:
            direction = 'ccw'

        # compute sweep-mean scan speed
        sweep_dt = time_vals[-1] - time_vals[0]
        if sweep_dt <= 0:
            print(f'Sweep {sweep}: invalid sweep duration, skipping')
            continue

        sweep_dtheta = np.abs(az_unwrapped[-1] - az_unwrapped[0])
        scan_speed = sweep_dtheta / sweep_dt

        speeds = np.array([10, 15, 20, 25])
        errors = np.array([0.042857, -0.657143, -1.328571, -2.042857])

        coef = np.polyfit(speeds, errors, 1)

        # correction = -error
        slope = -coef[0]
        intercept = -coef[1]

        correction = slope * scan_speed + intercept

        # apply sign based on sweep direction
        if direction == 'cw':
            radar.azimuth['data'][start:end] = (radar.azimuth['data'][start:end] + correction) % 360.0
            sign_str = '+'
        else:
            radar.azimuth['data'][start:end] = (radar.azimuth['data'][start:end] - correction) % 360.0
            sign_str = '-'

    return radar

def correct_azimuth_pointing_angle_sector(radar, offset=3.1, direction_sensitive=True, verbose=True):
    """
    Apply a simple hard-coded azimuth correction per sweep.
    This offset was found visiauuly and serves as a first pass, the correction may not be 100% accurate
    Parameters
    ----------
    radar : Py-ART radar object
    offset : float
        Azimuth correction magnitude in degrees.
    direction_sensitive : bool
        If True, apply +offset for clockwise sweeps and -offset for
        counterclockwise sweeps.
        If False, apply the same +offset to all sweeps.
    verbose : bool
        If True, print sweep-by-sweep correction info.

    Returns
    -------
    radar
        Py-ART radar object with azimuth corrections applied.
    """

    for sweep in range(radar.nsweeps):
        start = radar.sweep_start_ray_index['data'][sweep]
        end = radar.sweep_end_ray_index['data'][sweep] + 1

        az = np.asarray(radar.azimuth['data'][start:end])

        if len(az) < 2:
            continue

        # unwrap so crossing 360 does not fake a direction change
        az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(az)))

        # radar azimuth increasing = clockwise
        if az_unwrapped[-1] > az_unwrapped[0]:
            direction = 'cw'
        else:
            direction = 'ccw'

        if direction_sensitive:
            signed_offset = offset if direction == 'cw' else -offset
        else:
            signed_offset = offset

        radar.azimuth['data'][start:end] = (
            radar.azimuth['data'][start:end] + signed_offset
        ) % 360.0

        if verbose:
            print(
                f"Sweep {sweep}: direction={direction}, "
                f"applied correction={signed_offset:+.3f} deg"
            )

    return radar

def noise_filter(radar, field_in,  SNR = 5, rho = 0.6):
    '''
    Applies a basic clutter filter to filter out noise.
    It is based on thresholds of SNR and RHOHV, change these as necessary for your data.

    Parameters
    ----------
    radar: pyart radar object
    field_in: string 
        the field you wish to filter
    SNR: float 
        Signal to Noise Ratio Threshold, everything below this will be masked out, 5 is default
    rho: float
        Correlation Coefficient Threshold, everything below this will be masked out, 0.6 is default, which is a decent threshold for non-meteorological echos during convection

    Returns
    ------- 
        radar: pyart radar object with "F{field}" field added
    '''

    # applying thresholds to pyart gatefilter
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_below('SNR', SNR)
    gatefilter.exclude_below('RHO', rho)

    # Get reflectivity field
    field = radar.fields[field_in]
    field_data = field['data']

    #extracting field metadata
    coordinates = field['coordinates']
    valid_min = field['valid_min']
    valid_max = field['valid_max']
    standard_name = f"filtered_{field['standard_name']}"
    long_name = f"Filtered {field['long_name']}"
    units = field['units']

    # Mask data using gatefilter
    masked_data = np.ma.masked_where(gatefilter.gate_excluded, field_data)

    # Create new field dictionary
    filtered_field_dict = {
        'coordinates': coordinates,
        'valid_min': valid_min,
        'valid_max': valid_max,
        'standard_name': standard_name,
        'long_name': long_name,
        'units': units,
        'data': masked_data
        }

    # Add field to radar object
    radar.add_field(f'F{field_in}', filtered_field_dict, replace_existing=True)

    return radar

def find_files_in_timerange(root: Path, start_datetime: datetime, end_datetime: datetime) -> List[Path]:
    """
    Find all files in directories formatted as:

        root / 'YYYYMMDD' / 'ARMRYYYYMMDDHHMMSS.nc.xz'

    whose embedded filename timestamp is between start_datetime and end_datetime,
    inclusive.

    Parameters
    ----------
    root : Path
        Top-level directory containing YYYYMMDD subdirectories.
    start_datetime : datetime
        Start of desired time range.
    end_datetime : datetime
        End of desired time range.

    Returns
    -------
    List[Path]
        Sorted list of matching file paths.
    """

    if end_datetime < start_datetime:
        raise ValueError("end_datetime must be greater than or equal to start_datetime")

    matching_files = []

    # Start from the date part only
    current_date = start_datetime.date()
    final_date = end_datetime.date()

    while current_date <= final_date:
        date_str = current_date.strftime("%Y%m%d")
        day_dir = root / date_str

        if day_dir.exists() and day_dir.is_dir():
            for file_path in day_dir.glob("ARMR*.nc.xz"):
                try:
                    # Example filename: ARMR20260327230000.nc.xz
                    timestamp_str = file_path.name[len("ARMR"):len("ARMR") + 14]
                    file_datetime = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                except ValueError:
                    # Skip files that do not match expected naming format
                    continue

                if start_datetime <= file_datetime <= end_datetime:
                    matching_files.append(file_path)

        current_date += timedelta(days=1)

    return sorted(matching_files)

def filter_files_vcp(files, vcp_min, vcp_max):
    """
    Return a list of .xz CfRadial files whose VCP value falls within
    the interval [vcp_min, vcp_max).

    Parameters
    ----------
    files : iterable of str or Path
        Collection of `.xz` file paths to filter.

    vcp_min : int
        Minimum VCP value to retain (inclusive).

    vcp_max : int
        Maximum VCP value to retain (exclusive).

    Returns
    -------
    list of Path
        Files whose VCP values fall within the specified range.
    """
    kept_files = []

    for f in sorted(Path(f) for f in files):
        # decompress to temp .nc
        f_nc = decompress_xz(f)

        ds = xr.open_dataset(f_nc)
        try:
            vcp = int(ds.vcp_pattern)
        finally:
            ds.close()

        # keep file if within range
        if vcp_min <= vcp < vcp_max:
            kept_files.append(f)

        # remove the temporary .nc
        remove_nc(f_nc)

    return kept_files


def dealias_velocity(radar,vel_field='VEL',texture_field='velocity_texture',output_field='FVEL',wind_size=3,texture_threshold=3,centered=True):
    """
    Dealias Doppler velocity using a velocity-texture-based gatefilter.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART Radar object.

    vel_field : str, optional
        Name of the input velocity field. Default is 'VEL'.

    texture_field : str, optional
        Name of the velocity texture field to create/use. Default is
        'velocity_texture'.

    output_field : str, optional
        Name of the dealiased velocity field to save. Default is
        'corrected_velocity'.

    wind_size : int, optional
        Window size passed to Py-ART's velocity texture calculation.
        Default is 3.

    texture_threshold : float, optional
        Gates with velocity texture above this value will be excluded
        before dealiasing. Default is 3.

    centered : bool, optional
        Passed to pyart.correct.dealias_region_based. Default is True.

    Returns
    -------
    radar : pyart.core.Radar
        Radar object with:
        - texture_field added
        - output_field added
    """

    if vel_field not in radar.fields:
        raise KeyError(f"Field '{vel_field}' not found in radar.fields")

    if 'nyquist_velocity' not in radar.instrument_parameters:
        raise KeyError("Nyquist velocity not found in radar.instrument_parameters")

    nyquist = radar.instrument_parameters['nyquist_velocity']['data'][0]

    # Compute velocity texture
    vel_texture = pyart.retrieve.calculate_velocity_texture(
        radar,
        vel_field=vel_field,
        wind_size=wind_size,
        nyq=nyquist
    )

    radar.add_field(texture_field, vel_texture, replace_existing=True)

    # Build gatefilter from velocity texture
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_above(texture_field, texture_threshold)

    # Dealias velocity
    velocity_dealiased = pyart.correct.dealias_region_based(
        radar,
        vel_field=vel_field,
        nyquist_vel=nyquist,
        centered=centered,
        gatefilter=gatefilter
    )

    radar.add_field(output_field, velocity_dealiased, replace_existing=True)

    return radar


def radar_to_nc(radar,original_file,output_dir,suffix=None,overwrite=True):
    """
    Save a Py-ART radar object using the original ARMOR filename.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object to save.

    original_file : str or Path
        Original file path (used to extract filename).

    output_dir : str or Path
        Directory where the new file will be saved.

    suffix : str, optional
        Optional suffix to append to filename (e.g., '_qc').

    overwrite : bool, optional
        Whether to overwrite existing files.

    Returns
    -------
    Path
        Path to saved file.
    """

    original_file = Path(original_file)
    output_dir = Path(output_dir)

    # Get base filename (remove .xz if present)
    name = original_file.name
    if name.endswith('.xz'):
        name = name[:-3]  # remove '.xz'

    # Optionally add suffix before .nc
    if suffix:
        stem = Path(name).stem  # ARMR20260327...
        name = f"{stem}{suffix}.nc"

    output_path = output_dir / name

    # Make sure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle overwrite
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")

    # Save file
    pyart.io.write_cfradial(output_path, radar)

    return output_path