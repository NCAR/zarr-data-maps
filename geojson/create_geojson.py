#!/usr/bin/env python3
"""
Vectorize a raster mask from a NetCDF file into a GeoJSON border.

Requires: xarray, numpy, rasterio, shapely
  pip install xarray numpy rasterio shapely

Usage:
  python make_mask_border.py foo.nc -o foo_border.geojson
  # Options:
  #   --var MASKVAR        # name of mask variable (default: "regions")
  #   --region             # choose region_i to subset (default: 1)
  #   --invert             # treat 0/False as inside, 1/True as outside
  #   --polygon            # write filled Polygon/MultiPolygon instead of border
  #   --with-holes         # include interior holes in the border (ignored for --polygon)
  #   --simplify 0.01      # degrees; topology-preserving simplification
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from rasterio import features
from shapely.geometry import shape, mapping, MultiLineString
from shapely.ops import unary_union


def build_transform(lat_1d: np.ndarray, lon_1d: np.ndarray, north_up=True) -> Affine:
    """Affine transform mapping array rows/cols to geo coords at cell edges.

    If north_up=True, we assume row 0 is the *northernmost* row (common GIS).
    Our input latitudes are usually increasing south→north, so we’ll flip rows
    before polygonization and use a negative y pixel size.
    """
    # Nominal spacings (robust to tiny float jitter)
    dlon = float(np.median(np.diff(lon_1d)))
    dlat = float(np.median(np.diff(lat_1d)))
    if dlon <= 0 or dlat <= 0:
        raise ValueError("lat/lon must be strictly increasing 1-D arrays.")

    # Top-left corner at cell edge
    if north_up:
        yoff = float(lat_1d.max() + dlat / 2.0)  # top edge (north)
        xoff = float(lon_1d.min() - dlon / 2.0)  # left edge (west)
        return Affine(dlon, 0.0, xoff, 0.0, -dlat, yoff)  # negative y for north-up
    else:
        # South-up (uncommon; not used here)
        yoff = float(lat_1d.min() - dlat / 2.0)
        xoff = float(lon_1d.min() - dlon / 2.0)
        return Affine(dlon, 0.0, xoff, 0.0, dlat, yoff)


def to_minus180_180(lon_1d: np.ndarray) -> np.ndarray:
    """Convert longitudes to [-180, 180] if the series lives in [0,360]."""
    lon = lon_1d.astype(float).copy()
    if np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def main():
    p = argparse.ArgumentParser(description="Raster mask → GeoJSON border")
    p.add_argument("nc", help="Input NetCDF file path (e.g., foo.nc)")
    p.add_argument("-o", "--output", help="Output GeoJSON path (default: foo_border.geojson)")
    p.add_argument("--region", help="Region integer to select")
    p.add_argument("--var", default="regions", help='Mask variable name in the dataset (default: "regions")')
    p.add_argument("--invert", action="store_true", help="Invert mask (treat 0/False as inside)")
    p.add_argument("--polygon", action="store_true", help="Write filled Polygon/MultiPolygon instead of border")
    p.add_argument("--with-holes", action="store_true", help="Include hole boundaries in border (ignored with --polygon)")
    p.add_argument("--simplify", type=float, default=0.0, help="Simplify tolerance in degrees (default: 0 = no simplify)")
    args = p.parse_args()

    nc_path = Path(args.nc)
    if not nc_path.exists():
        sys.exit(f"Input not found: {nc_path}")

    region_i = (
        int(args.region)
        if args.region
        else 1
    )

    region_s = [
        'Empty',
        'North Atlantic',
        'Mid Atlantic',
        'Gulf Coast',
        'Pacific Northwest',
        'Pacific Southwest',
        'Northern Plains',
        'Mountain West',
        'Great Lakes',
        'Desert Southwest',

    ]
    print(f"Creating region {region_s[region_i]}")

    out_path = (
        Path(args.output)
        if args.output
        else Path(region_s[region_i].lower().replace(' ', '_')+'.geojson')
    )


    # 1) Open and locate variables
    ds = xr.open_dataset(nc_path)
    # Select region and turn it into a bool map
    ds['regions'] = ds['regions'].where(ds['regions']==region_i)
    ds['regions'] = ds['regions'].notnull()

    if "lat" not in ds.coords or "lon" not in ds.coords:
        sys.exit("Expected 1-D coordinates named 'lat' and 'lon'.")

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    lon = to_minus180_180(lon)  # normalize for GeoJSON consumers

    if args.var not in ds:
        sys.exit(f"Variable '{args.var}' not found in dataset.")

    m = ds[args.var]
    # Ensure dims are (lat, lon)
    if m.ndim != 2:
        sys.exit(f"Mask variable '{args.var}' must be 2-D; got shape {m.shape}")
    if m.dims != ("lat", "lon"):
        try:
            m = m.transpose("lat", "lon")
        except Exception:
            sys.exit(f"Mask variable '{args.var}' must be transposable to dims ('lat','lon'); got dims {m.dims}")

    mask_arr = m.values
    # Convert to boolean: True = inside region
    if mask_arr.dtype == bool:
        inside = mask_arr
    else:
        # Treat non-zero and non-NaN as inside
        inside = np.asarray(mask_arr) != 0
        inside &= ~np.isnan(mask_arr)

    if args.invert:
        inside = ~inside

    # 2) North-up orientation for polygonization: flip rows if lat increases south→north
    north_up = True
    lat_increasing = lat[1] > lat[0] if lat.size > 1 else True
    data_for_poly = inside[::-1, :] if (north_up and lat_increasing) else inside

    # 3) Build affine transform at cell edges
    transform = build_transform(lat, lon, north_up=True)

    # 4) Polygonize the "inside" class (value 1)
    data_uint8 = (data_for_poly.astype(np.uint8))  # 1 = inside, 0 = outside
    geoms = []
    for geom, val in features.shapes(data_uint8, transform=transform, connectivity=4):
        if int(val) == 1:
            geoms.append(shape(geom))
    if not geoms:
        sys.exit("No inside cells found in mask after processing.")

    # 5) Dissolve to a single (multi)polygon
    poly = unary_union(geoms)

    # Optional simplify
    if args.simplify and args.simplify > 0:
        poly = poly.simplify(args.simplify, preserve_topology=True)

    # 6) Convert to desired output geometry
    if args.polygon:
        out_geom = poly
        geom_type = out_geom.geom_type  # Polygon or MultiPolygon
    else:
        # Border (outer outline only by default)
        if poly.geom_type == "Polygon":
            lines = [poly.exterior]
            if args.with_holes:
                lines.extend(list(poly.interiors))
            out_geom = MultiLineString(lines) if len(lines) > 1 else lines[0]
        elif poly.geom_type == "MultiPolygon":
            exteriors = [p.exterior for p in poly.geoms]
            if args.with_holes:
                for p in poly.geoms:
                    exteriors.extend(list(p.interiors))
            out_geom = MultiLineString(exteriors) if len(exteriors) > 1 else exteriors[0]
        else:
            # Fallback: if union produced lines already (rare), pass through
            out_geom = poly
        geom_type = out_geom.geom_type  # LineString / MultiLineString

    # geojson needs to be LineString for the plotting to work
    mapping_out_geom = mapping(out_geom)
    if mapping_out_geom['type'] == 'LinearRing':
        mapping_out_geom['type'] = 'LineString'
    # print(mapping_out_geom)
    # print(mapping(out_geom)['type'])
    # sys.exit()
    # 7) Write GeoJSON
    feature = {
        "type": "Feature",
        "geometry": mapping_out_geom,
        "properties": {
            "source": str(nc_path),
            "variable": args.var,
            "type": "polygon" if args.polygon else "border",
            "simplify": args.simplify,
            "lon_domain": "[-180,180]",
        },
    }
    fc = {"type": "FeatureCollection", "features": [feature]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

    print(f"Wrote {geom_type} to {out_path}")


if __name__ == "__main__":
    main()
