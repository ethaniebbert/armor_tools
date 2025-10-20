'''
The purpose of this Python file is to serve as a place for all data analysis functions to live.
'''

import lzma
import gzip
from pathlib import Path
import os
import numpy as np

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