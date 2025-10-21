import matplotlib.pyplot as plt
import pyart
from datetime import datetime, timedelta
import numpy as np



def plot_rhi(file_path, fields, xmin = 0, xmax = 60, ymin = 0, ymax = 12, save_path = None):
    """
    Plot one or more fields from an RHI radar file.

    Parameters
    ----------
    file_path : str or Path
        Path to the .nc CFRadial radar file.
    fields : list of str
        List of field names to be plotted (e.g., ['reflectivity', 'velocity']).
    xmin, xmax : float, optional
        Horizontal range limits in km (default: 0–60).
    ymin, ymax : float, optional
        Vertical range limits in km (default: 0–12).
    save_path : str or Path, optional
        If provided, saves the figure to this file path.

    Returns
    -------
    None
        Displays or saves a figure containing one subplot per field.
    """

    # Reading in CFRadial File with Pyart
    radar = pyart.io.read(file_path)
    display = pyart.graph.RadarDisplay(radar)
    vnyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    sweeps = radar.sweep_number['data']

    # Dictionary of plotting parameters for each field
    FIELD_PARAMS = {
        'reflectivity': {'vmin': -10, 'vmax': 70, 'cmap': 'HomeyerRainbow', 'title': 'Reflectivity (dBZ)'},
        'velocity': {'vmin': -vnyq if vnyq is not None else -16,'vmax': vnyq if vnyq is not None else 16,'cmap': 'PuOr_r', 'title': 'Radial Velocity (m/s)'},
        'spectrum_width': {'vmin': 0, 'vmax': 10, 'cmap': 'pyart_NWS_SPW', 'title': 'Spectrum Width (m/s)'},
        'cross_correlation_ratio': {'vmin': 0.4, 'vmax': 1.05, 'cmap': 'plasmidis', 'title': 'RHO (ρhv)'},
        'differential_reflectivity': {'vmin': -2, 'vmax': 6, 'cmap': 'pyart_RefDiff', 'title': 'ZDR (dB)'},
    }

    # Extract base time string
    units_str = radar.time['units']
    base_time_str = units_str.split('since')[-1].strip().replace('Z', '')
    base_time = datetime.fromisoformat(base_time_str)

    for snum in sweeps:
        # getting metadata for each sweep
        sweep_starts = radar.sweep_start_ray_index['data'][snum]
        sweep_ends = radar.sweep_end_ray_index['data'][snum]
        azimuth = radar.azimuth['data'][sweep_starts]
        sweep_time_seconds = round(radar.time['data'][sweep_starts])
        sweep_datetime = base_time + timedelta(seconds=float(sweep_time_seconds))
        sweep_time = f"{sweep_datetime.isoformat()}Z"


        # plotting
        nplots = len(fields)
        fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 5))
        axes = np.atleast_1d(axes).flatten()

        for i, field in enumerate(fields):
            if field not in FIELD_PARAMS:
                print(f"Warning: Unknown field '{field}', skipping.")
                continue

            params = FIELD_PARAMS[field]
            ax = axes[i]

            display.plot_rhi(
                field, sweep=snum, ax=ax,
                vmin=params['vmin'], vmax=params['vmax'],
                cmap=params['cmap'],
                colorbar_label=params['title'],
                title=f"ARMOR RHI | {params['title']}  \n  Az: {azimuth:.1f}°  |  {sweep_time}"
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('Distance from Radar (km)')
            ax.set_ylabel('Height (km)')

            # Turn on major and minor ticks
            ax.minorticks_on()
            # Major gridlines
            ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
            # Minor gridlines
            ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        # getting rid of "bad" filename characters for time string
        safe_time = (
            sweep_time.replace(":", "")
            .replace("T", "")
            .replace("Z", "")
        )

        # if save path is specified, saves the figure, if not then displays the figure
        if save_path:
            out_path = Path(save_path) / f"ARMR_RHI_{safe_time}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()