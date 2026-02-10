import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from netCDF4 import Dataset


def cmap_from_png(png_path, n=256, sample_band=5, tick_white_thresh=0.98):
    """
    Build a matplotlib colormap from a horizontal colorbar PNG.

    png_path: path to the PNG
    n: number of colors in the resulting colormap
    sample_band: number of rows around the vertical midpoint to average (reduces noise)
    tick_white_thresh: threshold for detecting near-white tick-mark pixels (0–1)
    """
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3) in [0,1]

    H, W, _ = arr.shape

    # Sample around the vertical midpoint (avoids bottom tick marks in many images)
    y0 = max(0, H // 2 - sample_band // 2)
    y1 = min(H, y0 + sample_band)
    strip = arr[y0:y1, :, :].mean(axis=0)             # (W, 3)

    # If any near-white columns exist (tick marks), patch them by interpolation
    is_tick = np.all(strip > tick_white_thresh, axis=1)  # (W,)
    if np.any(is_tick):
        x = np.arange(W)
        good = ~is_tick
        # Only interpolate if we still have enough non-tick pixels
        if good.sum() >= 2:
            for c in range(3):
                strip[:, c] = np.interp(x, x[good], strip[good, c])

    # Resample to n colors
    x = np.arange(W)
    xi = np.linspace(0, W - 1, n)
    lut = np.vstack([np.interp(xi, x, strip[:, c]) for c in range(3)]).T  # (n,3)

    return ListedColormap(lut, name="cmap_from_png")


def read_wrf_2d_at_t0(nc_path, varname="PSFC", latname="XLAT", lonname="XLONG"):
    """
    Read WRF 2D variables with dims (Time, south_north, west_east).
    Returns lon2d, lat2d, var2d as float64 arrays.
    """
    with Dataset(nc_path, "r") as ds:
        for name in [varname, latname, lonname]:
            if name not in ds.variables:
                raise KeyError(f'Variable "{name}" not found in {nc_path}')

        var = ds.variables[varname]
        lat = ds.variables[latname]
        lon = ds.variables[lonname]

        # Take Time=0 and squeeze to (south_north, west_east)
        var2d = np.squeeze(var[0, :, :]).astype(np.float64)
        lat2d = np.squeeze(lat[0, :, :]).astype(np.float64)
        lon2d = np.squeeze(lon[0, :, :]).astype(np.float64)

        # Optional: carry units if present
        units = getattr(var, "units", None)

    return lon2d, lat2d, var2d, units


def plot_psfc_contourf(lon2d, lat2d, psfc2d, cmap, units=None, levels=40, title=None):
    fig, ax = plt.subplots(figsize=(9, 6))

    # If you prefer explicit levels:
    # levels = np.linspace(np.nanmin(psfc2d), np.nanmax(psfc2d), 41)

    cf = ax.contourf(lon2d, lat2d, psfc2d, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar_label = "PSFC" + (f" [{units}]" if units else "")
    cbar.set_label(cbar_label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title if title else "WRF PSFC")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wrf_path = "/glade/derecho/scratch/rozoff/wrf/delta/sim3/wrfout_d03_2020-10-09_06:00:00"
    png_path = "ncview_choices/3gauss.png"

    # Build colormap from PNG
    cmap = cmap_from_png(png_path, n=256, sample_band=5, tick_white_thresh=0.98)

    # Read WRF fields (Time=0)
    lon2d, lat2d, psfc2d, units = read_wrf_2d_at_t0(wrf_path, varname="PSFC", latname="XLAT", lonname="XLONG")

    # Plot
    plot_psfc_contourf(
        lon2d, lat2d, psfc2d,
        cmap=cmap,
        units=units,
        levels=256,
        title="PSFC from wrfout_d03_2020-10-09_06:00:00 (t=0)"
    )