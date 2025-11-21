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

def read_nc_pyart(input_file):
    radar = pyart.io.read(input_file)
    return radar

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
    Filter .xz CfRadial files by VCP pattern.

    Only files whose VCP value falls within the inclusive–exclusive
    interval [vcp_min, vcp_max) are kept. Files with VCP values
    outside this range are removed.

    Common VCP ranges
    -----------------
       0–99 : RHI scans
    100–199 : Sector scans
    200–299 : Full-volume PPI scans
    300-399 :
    400–499 : Calibration scans

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
    None
        Files are removed in place. No value is returned.
    """
    file_path = Path(file_folder)
    files = sorted(file_path.glob("*.xz"))

    for f in files:
        f_nc = decompress_xz(f)
        ds = xr.open_dataset(f_nc)
        try:
            vcp = int(ds.vcp_pattern)
        finally:
            ds.close()

        # Delete if outside desired range
        if not (vcp_min <= vcp < vcp_max):
            os.remove(f)

        remove_nc(f_nc)