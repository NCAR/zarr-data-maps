import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import xarray as xr
import xesmf as xesmf
import importlib
# import funcs_esmeval as fesm
# copied from /glade/u/home/nlybarger/scripts/python_funcs
import python_funcs.funcs_esmeval as fesm
import argparse
import sys

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Example script that accepts --input, --output, and --region."
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input file"
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output file"
    )
    p.add_argument(
        "--scenario", "-s",
        required=True,
        help="Scenario (e.g., 'historical', 'rcp45', 'rcp85')"
    )
    p.add_argument(
        "--cmip", "-c",
        required=True,
        help="cmip choice"
    )
    p.add_argument(
        "--region", "-r",
        default="global",
        help="Region name/ID (e.g., 'conus', 'frontrange', 'us-west')"
    )
    return p.parse_args()

args = parse_args()
print("arguments:", args)
# diri = f'/glade/u/home/nlybarger/scratch/data/climate_data/{cmip}/postproc/'
# diro = f'/glade/work/nlybarger/data/esmeval/{cmip}_metrics/NCA_regions/{dom}/'sys.exit()
diri = args.input + '/' + args.scenario

print("input case directory", diri)

# Reading in CMIP datasets and parsing model/variant information
nca_regions = ['Northeast', 'Southeast', 'Midwest', 'Northern Great Plains',
               'Southern Great Plains', 'Southwest', 'Northwest']

# not used
# odiri = '/glade/campaign/ral/hap/nlybarger/OBS/'

metrics_allobs = {}
metrics_obsmean = {}
model_metrics = {}
# odsets = ['CRU','ERA-5','UDel','Livneh','PRISM']
# fnames = ['cru','era5','udel','livneh','prism']
# odsets = ['CRU','ERA-5','UDel']
# fnames = ['cru','era5','udel']
cmips = ['cmip5','cmip6']

regind = next((i for i, v in enumerate(nca_regions) if v == args.region), None)
cmipind = cmips.index(args.cmip)
cmip = cmips[cmipind]
if cmip in diri:
    print(f"cmip {cmip} in diri")
else:
    print(f"cmip {cmip} not in diri")
    sys.exit()
if regind != None:
    region = nca_regions[regind]
    odsets = ['CRU','ERA5','UDel','Livneh','PRISM']
    fnames = ['cru','era5','udel','livneh','prism']
else:
    region = 'global'
    odsets = ['CRU','ERA5','UDel']
    fnames = ['cru','era5','udel']

print("regio =", region)
OVERWRITE = True
# for region in nca_regions:
if region in nca_regions:
    obs, oyears, oseasvars = fesm.read_obs_climate_data(odsets, fnames, region)
    metrics_allobs[region], metrics_obsmean[region] = \
        fesm.compute_obs_metrics(obs, odsets, oyears, oseasvars, region)

    # todo diri needs to be updated to campagin and diro needs to be changed
    # to mine

    # diri = f'/glade/u/home/nlybarger/scratch/data/climate_data/{cmip}/postproc/'
    filis = sorted(glob.glob(diri+'*historical.nc'))
    nfil = len(filis)
    models = ['0']*nfil
    for i in range(nfil):
        models[i] = filis[i].split('/')[-1].split('_')[0].split('.')[0]
    for imod, mod in enumerate(models):
        dom = region.replace(' ','')
        # diro = f'/glade/work/nlybarger/data/esmeval/{cmip}_metrics/NCA_regions/{dom}/'
        diro = f'{args.output}/{cmip}_metrics/NCA_regions/{dom}/'
        print("output directory", diro)
        filo = f'{mod}.{cmip}.metrics.{dom}.pkl'

        if (os.path.exists(f'{diro}{filo}')) and (not OVERWRITE):
            print(f'Metrics file for {mod} already exists. Loading...')
            continue
        else:
            nci, variants = fesm.read_cmip_data(filis[imod], region)
            _ = fesm.compute_model_metrics(nci, mod, variants, region, oseasvars, metrics_allobs[region], cmip, OVERWRITE=OVERWRITE)
            del nci
            del variants
elif region == 'global':
    obs, oyears, oseasvars = fesm.read_obs_climate_data(odsets, fnames, region)
    metrics_allobs, metrics_obsmean = fesm.compute_obs_metrics(obs, odsets, oyears, oseasvars, region)

    # diri = f'/glade/u/home/nlybarger/scratch/data/climate_data/{cmip}/postproc/'
    filis = sorted(glob.glob(diri+'*historical.nc'))
    nfil = len(filis)
    models = ['0']*nfil
    for i in range(nfil):
        models[i] = filis[i].split('/')[-1].split('_')[0].split('.')[0]


    # for imod, mod in enumerate(models):
    imod = 0
    mod = models[imod]
    dom = region.replace(' ','')
    # diro = f'/glade/work/nlybarger/data/esmeval/{cmip}_metrics/{dom}/'
    diro = f'{args.output}/{cmip}_metrics/{dom}/'
    filo = f'{mod}.{cmip}.metrics.{dom}.pkl'

    if (os.path.exists(f'{diro}{filo}')) and (not OVERWRITE):
        print(f'Metrics file for {mod} already exists. Loading...')
        # continue
    else:
        nci, variants = fesm.read_cmip_data(filis[imod], region)
        _ = fesm.compute_model_metrics(nci, mod, variants, region, oseasvars, metrics_allobs, cmip, OVERWRITE=OVERWRITE)
        del nci
        del variants
