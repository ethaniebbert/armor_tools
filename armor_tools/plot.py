import matplotlib.pyplot as plt
import pyart
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path



def plot_rhi(radar, fields, xmin = 0, xmax = 60, ymin = 0, ymax = 12, save_path = None, grids = False):
    """
    Generate Range–Height Indicator (RHI) plots for one or more radar fields.

    This function creates one subplot per requested field from an RHI scan,
    applying field-specific colormaps, limits, and titles. Each sweep is plotted
    individually with metadata such as azimuth angle and sweep time. Figures can
    be displayed interactively or saved to disk.

    Parameters
    ----------
    radar : pyart.core.Radar
        PyART Radar object
    fields : list of str
        Names of fields to plot (e.g., ['reflectivity', 'velocity']).
    xmin, xmax : float, optional
        Horizontal range limits in km. Defaults to 0–60 km.
    ymin, ymax : float, optional
        Vertical range limits in km. Defaults to 0–12 km.
    save_path : str or Path, optional
        Directory where output figures will be written. If None, figures display
        instead of saving.
    grids : bool, optional
        If True, enables major and minor gridlines on each subplot.

    Returns
    -------
    None
        Displays or saves RHI figures for each sweep.
    """

    # Creates display object
    display = pyart.graph.RadarDisplay(radar)
    vnyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    sweeps = radar.sweep_number['data']

    # Dictionary of plotting parameters for each field
    FIELD_PARAMS = {
        # Reflectivity
        'reflectivity': {'vmin': -10, 'vmax': 70, 'cmap': 'HomeyerRainbow', 'title': 'Reflectivity (dBZ)'},
        'REF': {'vmin': -10, 'vmax': 70, 'cmap': 'HomeyerRainbow', 'title': 'Reflectivity (dBZ)'},

        # Velocity
        'velocity': {'vmin': -16, 'vmax': 16, 'cmap': 'PuOr_r', 'title': 'Radial Velocity (m/s)'},
        'VEL': {'vmin': -16, 'vmax': 16, 'cmap': 'PuOr_r', 'title': 'Radial Velocity (m/s)'},

        # Differential Reflectivity
        'differential_reflectivity': {'vmin': -2, 'vmax': 6, 'cmap': 'ChaseSpectral', 'title': 'ZDR (dB)'},
        'ZDR': {'vmin': -2, 'vmax': 6, 'cmap': 'ChaseSpectral', 'title': 'ZDR (dB)'},

        # Cross-correlation ratio
        'cross_correlation_ratio': {'vmin': 0.4, 'vmax': 1.05, 'cmap': 'plasma', 'title': 'RHO (ρhv)'},
        'RHO': {'vmin': 0.4, 'vmax': 1.05, 'cmap': 'plasma', 'title': 'RHO (ρhv)'},

        # Spectrum width
        'spectrum_width': {'vmin': 0, 'vmax': 10, 'cmap': 'pyart_NWS_SPW', 'title': 'Spectrum Width (m/s)'},
        'SW': {'vmin': 0, 'vmax': 10, 'cmap': 'pyart_NWS_SPW', 'title': 'Spectrum Width (m/s)'},
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

            # Turn on major and minor ticks if True
            if grids == True:
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


def plot_ppi(radar, fields, sweeps = False, xmin=-60, xmax=60, ymin=-60, ymax = 60, save_path = None, grids = False):
    """
        Generate Plan Position Indicator (PPI) plots for one or more radar fields.

        This function creates one subplot per requested field from specified PPI
        sweeps. Each panel includes field‐specific colormaps, limits, and titles,
        along with sweep metadata such as elevation angle and timestamp. Figures can
        either be displayed interactively or saved to a directory.

        Parameters
        ----------
        radar : pyart.core.Radar
            PyART Radar object.
        fields : list of str
            Names of fields to plot (e.g., ['reflectivity', 'velocity']).
        sweeps : list of int or bool, optional
            Sweep numbers to plot. If False (default), all sweeps available in the
            radar file are plotted.
        xmin, xmax : float, optional
            Horizontal plot extent in km (east–west). Defaults to –60 to 60 km.
        ymin, ymax : float, optional
            Vertical plot extent in km (north–south). Defaults to –60 to 60 km.
        save_path : str or Path, optional
            Directory where output figures will be written. If None, figures display
            instead of saving.
        grids : bool, optional
            If True, enables major and minor gridlines on each subplot.

        Returns
        -------
        None
            Displays or saves PPI figures for each sweep.
        """
    # Creates display object
    display = pyart.graph.RadarDisplay(radar)
    vnyq = radar.instrument_parameters['nyquist_velocity']['data'][0]

    if sweeps == False:
        sweeps = radar.sweep_number['data']
    else:
        sweeps = sweeps

    # Dictionary of plotting parameters for each field
    FIELD_PARAMS = {
        # Reflectivity
        'reflectivity': {'vmin': -10, 'vmax': 70, 'cmap': 'HomeyerRainbow', 'title': 'Reflectivity (dBZ)'},
        'REF': {'vmin': -10, 'vmax': 70, 'cmap': 'HomeyerRainbow', 'title': 'Reflectivity (dBZ)'},

        # Velocity
        'velocity': {'vmin': -16, 'vmax': 16, 'cmap': 'PuOr_r', 'title': 'Radial Velocity (m/s)'},
        'VEL': {'vmin': -16, 'vmax': 16, 'cmap': 'PuOr_r', 'title': 'Radial Velocity (m/s)'},

        # Differential Reflectivity
        'differential_reflectivity': {'vmin': -2, 'vmax': 6, 'cmap': 'ChaseSpectral', 'title': 'ZDR (dB)'},
        'ZDR': {'vmin': -2, 'vmax': 6, 'cmap': 'ChaseSpectral', 'title': 'ZDR (dB)'},

        # Cross-correlation ratio
        'cross_correlation_ratio': {'vmin': 0.4, 'vmax': 1.05, 'cmap': 'plasma', 'title': 'RHO (ρhv)'},
        'RHO': {'vmin': 0.4, 'vmax': 1.05, 'cmap': 'plasma', 'title': 'RHO (ρhv)'},

        # Spectrum width
        'spectrum_width': {'vmin': 0, 'vmax': 10, 'cmap': 'pyart_NWS_SPW', 'title': 'Spectrum Width (m/s)'},
        'SW': {'vmin': 0, 'vmax': 10, 'cmap': 'pyart_NWS_SPW', 'title': 'Spectrum Width (m/s)'},
    }

    # Extract base time string
    units_str = radar.time['units']
    base_time_str = units_str.split('since')[-1].strip().replace('Z', '')
    base_time = datetime.fromisoformat(base_time_str)

    for snum in sweeps:
        # getting metadata for each sweep
        sweep_starts = radar.sweep_start_ray_index['data'][snum]
        sweep_ends = radar.sweep_end_ray_index['data'][snum]
        elevation = radar.elevation['data'][sweep_starts]
        sweep_time_seconds = round(radar.time['data'][sweep_starts])
        sweep_datetime = base_time + timedelta(seconds=float(sweep_time_seconds))
        sweep_time = f"{sweep_datetime.isoformat()}Z"


        # plotting
        nplots = len(fields)
        fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 5))
        axes = np.atleast_1d(axes).flatten()

        for i, field in enumerate(fields):

            params = FIELD_PARAMS[field]
            ax = axes[i]

            display.plot_ppi(
                field, sweep=snum, ax=ax,
                vmin=params['vmin'], vmax=params['vmax'],
                cmap=params['cmap'],
                colorbar_label=params['title'],
                title=f"ARMOR PPI | {params['title']}  \n  El: {elevation:.1f}°  |  {sweep_time}"
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('Distance from Radar (E/W) (km)')
            ax.set_ylabel('Distance from Radar (N/S) (km)')

            # Turn on major and minor ticks if True
            if grids == True:
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
            out_path = Path(save_path) / f"ARMR_PPI_{safe_time}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()