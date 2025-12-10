import armor_tools.analysis as an
from pathlib import Path
import pyart

# === Setup ===
# update root to the directory path containing your data
root = Path('/Users/eebbert/Downloads/example_data/')

# Standard data directory (adjust only if needed)
DATA = root / 'analysis_data1'

# Example input files
sample_xz = DATA / "ARMR20251205093641.nc.xz"
sample_gz = DATA / "ARMR20251205093641_RHI.nc.gz"
sample_nc = DATA / "ARMR20251205093542_RHI.nc"
sample_L2 = DATA / "ARMR20251205094011"

# Output directory for converted files
OUTPUT = root / "analysis_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


# === Decompressing Radar Files ===
'''
Workflow:
Pick a compressed radar file (.xz or .gz)
Decompress it into a temporary .nc file
Use it however you want (e.g., read into Py-ART)
Remove the temporary file afterward
'''

# Decompress .xz → .nc
temp_nc_from_xz = an.decompress_xz(sample_xz)
print("Decompressed .xz to:", temp_nc_from_xz)

# Decompress .gz → .nc
temp_nc_from_gz = an.decompress_gz(sample_gz)
print("Decompressed .gz to:", temp_nc_from_gz)

# (Optional) read into Py-ART
radar_xz = pyart.io.read(temp_nc_from_xz)
radar_gz = pyart.io.read(temp_nc_from_gz)

# Remove temporary files
an.remove_nc(temp_nc_from_xz)
an.remove_nc(temp_nc_from_gz)


# === Listing Fields Inside a CfRadial File ===
'''
Workflow:
Open a CfRadial .nc file
List all internal radar fields
Useful for choosing which to plot
'''

fields = an.list_fields(sample_nc)
print("Available fields in file:")
for f in fields:
    print(" -", f)


# === Converting Level II → CfRadial ===
'''
Workflow:
Take a Level II archive file
Convert it to Py-ART-compatible CfRadial format
Save the result to an output directory
'''

cfrad_path = an.L2_to_CFRad(sample_L2, OUTPUT)
print("Converted Level II file saved to:", cfrad_path)


# === Beam Height Calculation Example ===
'''
Workflow:
Provide range (m) + elevation angle (deg)
Compute beam height above radar level
'''

rng_m = 30000     # 30 km
elev_deg = 2.0
height = an.cal_beam_height(rng_m, elev_deg)
print(f"Beam height at {rng_m/1000} km and {elev_deg}°:", height, "m")


# === Elevation Angle Calculation Example ===
'''
Workflow:
Provide range (m) + height above radar level (m)
Solve for the elevation angle required
'''

target_height = 3500   # meters
required_elev = an.cal_elev_angle(rng_m, target_height)
print(f"Elevation angle for {target_height} m height:", required_elev, "degrees")