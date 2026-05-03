#!/usr/bin/env python3

from pathlib import Path
import json
import xarray as xr


input_dir = Path(".")
output_file = Path("agreement_variables.js")

agreement_variables = {}

for path in sorted(input_dir.glob("*signal*.nc")):
    print(f"Reading {path.name}")
    if path.name in ['GARDLENS_CMIP6_ssp370_signal_pr.nc',
                     'GARDLENS_CMIP6_ssp370_signal_tasmax.nc',
                     'GARDLENS_CMIP6_ssp370_signal_tasmin.nc',
                     ]:
        print(f"Skipping {path.name}")
        continue

    parts = path.stem.split("_")
    # Examples:
    #   BCCA_CMIP5_signal_pr
    #   GARDLENS_CMIP6_ssp370_signal_tasmax

    downscaling = parts[0].lower().replace('-','_')
    cmip = parts[1].lower().replace('-','_')

    if parts[2] == "signal":
        scenario = ""
        climate_variable = parts[3].lower().replace('-','_')
    else:
        scenario = parts[2].lower()
        climate_variable = parts[-1].lower().replace('-','_')

    with xr.open_dataset(path) as ds:
        if "gcm" not in ds:
            print(f"  skipping {path.name}: no gcm variable")
            continue

        for gcm in ds["gcm"].values:
            model = str(gcm).lower().replace('-','_')

            variables = list(ds.sel(gcm=gcm).variables)

            for coord_name in ["lat", "lon", "gcm"]:
                if coord_name in variables:
                    variables.remove(coord_name)

            agreement_variables.setdefault(cmip, {})
            agreement_variables[cmip].setdefault(downscaling, {})
            agreement_variables[cmip][downscaling].setdefault(model, {})
            agreement_variables[cmip][downscaling][model].setdefault(scenario, {})
            agreement_variables[cmip][downscaling][model][scenario].setdefault(climate_variable, {})
            agreement_variables[cmip][downscaling][model][scenario][climate_variable].setdefault("variables", [])
            existing = agreement_variables[cmip][downscaling][model][scenario][climate_variable]["variables"]

            for variable in variables:
                if variable not in existing:
                    existing.append(variable)

            existing.sort()

js_text = f"""// Auto-generated. Do not edit manually.

export const agreement_variables = {json.dumps(agreement_variables, indent=2)};
"""

output_file.write_text(js_text)

print(f"Wrote {output_file}")
