from itertools import combinations
import re
import warnings
import numpy as np
import xarray as xr
from Functions_Extreme_WTs import PreprocessWTdata, ClusterAnalysis, EucledianDistance
import funcs_DS_eval as fdse
import glob
import pickle
import os
import funcs_read_ds_cmip5 as frdc5
import funcs_read_obs as fro
import funcs_general as fg
import xskillscore as xs
import datetime as dt
import pandas as pd
import xesmf
import cftime


def load_raw(model):
    diri = '/glade/campaign/ral/hap/nlybarger/cmip5_preproc_for_WT/'
    if model != 'MIROC5':
        nci = xr.open_dataset(f'{diri}{model}.historical_rcp45.1981-2016.WTpreproc.nc')
    else:
        nci = xr.open_dataset(f'{diri}{model}.historical_rcp85.1981-2016.WTpreproc.nc')

    vardict = dict({'Q': 'hus', 'U': 'ua', 'V': 'va', 'UV': 'uva',
                'ZG': 'zg', 'MFL': 'qflux',
                'PSL': 'psl', 'PW': 'pw'})
    return nci, vardict


def varlev(str):
# Separates variable string into Variable name and float level in mb
    var = re.findall(r'[a-zA-Z]+', str)
    levstr = re.findall(r'[0-9]+', str)
    if levstr:
        lev = float(levstr[0])*100.
    else:
        lev = None
    return var[0],lev


def get_regional_WT_varlist():
    VariableList = {}
    VariableList['North Atlantic'] = ['V850', 'Q500', 'ZG500', 'PW', 'PSL']
    VariableList['Mid Atlantic'] = ['UV850', 'Q500', 'ZG500', 'MFL500', 'PW']
    VariableList['Gulf Coast'] = ['V850', 'V250', 'UV850', 'Q850', 'Q500']
    VariableList['Pacific Northwest'] = ['V500', 'UV850', 'Q500', 'ZG500', 'PSL']
    VariableList['Pacific Southwest'] = ['UV850', 'UV500', 'PSL']
    VariableList['Northern Plains'] = ['V850', 'V250', 'Q500', 'MFL500']
    VariableList['Mountain West'] = ['V850', 'UV850', 'Q850', 'Q500']
    VariableList['Great Lakes'] = ['UV850', 'Q500', 'MFL500', 'PW', 'PSL']
    VariableList['Desert Southwest'] = ['U500', 'UV850', 'MFL500', 'PW', 'PSL']
    regions = list(VariableList.keys())
    return regions, VariableList


def create_norm_array(mod, TimeStart, TimeEnd, latrng, lonrng, RegionName, VariableList, ens=''):
    timslic = slice(TimeStart, TimeEnd)
    print('Beginning to compute WT centroids for model: ' + mod +
          ', over the period: ' + TimeStart[0:4] + '-' + TimeEnd[0:4] +
          ', for the region: ' + RegionName)

    print('|===| Reading data')

    if mod == 'era5':
        nci = fro.read_obs_daily('ERA5', timslic).sel(lat=latrng, lon=lonrng).load()
    elif mod == 'CESM2-LE':
        if ens == '':
            print('CESM2 LE requires ensemble member to be specified')
            return
        else:
            diri = '/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/'
            fili = f'{diri}CESM2-LE_WTvars_CONUS_{ens}_1981-2016.hist_1deg.nc'
            nci = xr.open_dataset(fili, engine='netcdf4').sel(time=timslic, lat=latrng, lon=lonrng).load()
        nci = nci.rename({'TMQ':'PW'})
        nci = nci.rename({'U200':'U250', 'V200':'V250', 'Z200':'Z250', 'Q200':'Q250', 'T200':'T250'})
        for lev in ['850', '700', '500', '250']:
            nci['UV' + lev] = np.sqrt(np.square(nci['U' + lev]) + np.square(nci['V' + lev]))
            nci['MFL' + lev] = nci['Q' + lev] * nci['UV' + lev]
            nci = nci.rename({f'Z{lev}':f'ZG{lev}'})
    else:
        nci, vardict = load_raw(mod)
        nci = nci.sel(time=timslic, lat=latrng, lon=lonrng,drop=True)
        nci['pr'] = nci['pr']*86400.

    LonWT = nci['lon']
    LatWT = nci['lat']
    nlon = len(LonWT)
    nlat = len(LatWT)

    rdthresh = (xr.where(nci['pr'] > 1., 1, 0)).sum(dim=['lat', 'lon'])
    rainy_days = nci.where(rdthresh > ((nlat*nlon)*0.25))
    del nci
    del rdthresh
    rdinds = np.where(~np.isnan(rainy_days['pr'].mean(dim=['lat', 'lon']).values))[0]
    rainy_days = rainy_days.dropna(dim='time', how='all')
    rdpr = rainy_days['pr']
    rdntim = rdpr.shape[0]

    print('|===| Casting data to numpy array')
    nvar = len(VariableList)
    nciv = np.zeros((rdntim,nlat,nlon,nvar))

    for ivar, var in enumerate(VariableList):
        
        if mod == 'era5':
            varname = var
            vardat = rainy_days[varname].copy()
        elif mod != 'CESM2-LE':
            tmpvar, tmplev = varlev(var)
            if tmpvar in list(vardict.keys()):
                varname = vardict[tmpvar]
                if tmplev:
                    vardat = rainy_days[varname].copy().sel(plev=tmplev, method='nearest')
                else:
                    vardat = rainy_days[varname].copy()
            else:
                print('Variable ' + tmpvar + ' not found for model: ' + mod)
        else:
            varname = var
            vardat = rainy_days[varname].copy()

        nciv[:,:,:,ivar] = vardat.data
        del vardat
    del rainy_days

    print('|===| Beginning to normalize variables')
    dvmean, dvstd, varsNorm = PreprocessWTdata(nciv)
    del nciv
    return dvmean, dvstd, varsNorm, rdpr, rdntim, rdinds


def combine_norm_arrays(ntim, pr, varsNorm, nlat, nlon, nvar):
    allntim = 0

    for i in list(ntim.keys()):
        allntim += ntim[i]

    varsNormAll = np.zeros((allntim, nlat, nlon, nvar))
    prall = np.zeros((allntim, nlat, nlon))
    strt = 0
    mods = list(varsNorm.keys())
    for i in range(len(mods)):
        if i == 0:
            endd = ntim[mods[i]]
        varsNormAll[strt:endd,:,:,:] = varsNorm[mods[i]]
        prall[strt:endd,:,:] = pr[mods[i]].data
        strt += ntim[mods[i]]
        if i < len(mods)-1:
            endd += ntim[mods[i+1]]
    return varsNormAll, prall


def categorize_precip_WT(regname, varsNorm, rdntim, numWT=6):
    with open(f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/norm_factors/{regname}_rgrClustersFin.pkl', 'rb') as file:
        rgrClustersFin = pickle.load(file)
    EucledianDist, _ = EucledianDistance(varsNorm, rgrClustersFin)
    tmp = np.zeros(rdntim, dtype=int)
    for i in range(rdntim):
        tmp[i] = int(np.argmin(EucledianDist[i,:]))
    
    WT_index = {}
    for i in range(numWT):
        wt = str(i)
        WT_index[wt] = np.where(tmp == i)[0]
    return WT_index


def remove_leap_days(prarray):
    wtpr = prarray.sel(time=~((prarray.time.dt.month == 2) & (prarray.time.dt.day == 29)), drop=True)
    return wtpr


def load_WT_centroids(regname, varstr, TimeStart='1981-01-01', TimeEnd='2016-12-31'):
    diri = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/'
    centroids = xr.open_dataset(f'{diri}{regname}/{regname}.era5.WT.centroids.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.nc')
    WT_index = np.load(f'{diri}{regname}/era5.{regname}.index.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.npy', allow_pickle=True).item()
    mWTprecip = np.load(f'{diri}{regname}/era5.{regname}.precip.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.npy', allow_pickle=True)

    return WT_index, mWTprecip, centroids


def Compute_WT_centroids_ERA5(TimeStart, TimeEnd, VariableList, lonrng, latrng, RegionName, numberWTs=0, OVERWRITE=False):
    varstr = ''
    regname = RegionName.replace(' ', '')
    for var in VariableList:
        varstr += var+'.'
    diro = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/'
    filo = f'{regname}/{regname}.era5.WT.centroids.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.nc'

    if OVERWRITE:
        if os.path.exists(diro+filo):
            print('Designated centroids already exist.')
            print('Overwrite mode active. Removing nc file')
            os.remove(diro+filo)
    else:
        if os.path.exists(diro+filo):
            print('Weather type centroids already computed for this time period and region.  Continuing without computing.')
            return

    dvmean, dvstd, varsNorm, rdpr, rdntim, rdinds = \
    create_norm_array('era5', TimeStart, TimeEnd, latrng, lonrng, RegionName, VariableList)
    LonWT = rdpr['lon']
    nlon = len(LonWT)
    LatWT = rdpr['lat']
    nlat = len(LatWT)

    nvar = len(VariableList)

    print('Running cluster analysis')
    rgrClustersFin=ClusterAnalysis(varsNorm, diro, 0, TimeStart[0:4]+'-'+TimeEnd[0:4], Plot=0, numWT=numberWTs)
    CentroidsAct=rgrClustersFin[0]
    numWTs = CentroidsAct.shape[0]
    WTfreq = np.zeros(numWTs)

    print('|===| Beginning to partition precipitation by WT')
    WTprecip = np.zeros((numWTs, nlat, nlon))
    WT_index = {}
    fullinds = {}
    for i in range(numWTs):
        wt = str(i)
        WT_index[wt] = np.where(rgrClustersFin[1] == i)[0]
        tmp = WT_index[wt]
        print(tmp)

        WTfreq[i] = round((np.count_nonzero(rgrClustersFin[1] == i)/rdntim)*100, 1)
        WTprecip[i,:,:] = np.mean(rdpr[tmp,:,:], axis=0)
        fullinds[wt] = rdinds[WT_index[wt]]

    index_fout = f'{diro}/{regname}/era5.{regname}.index.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.npy'
    preci_fout = f'{diro}/{regname}/era5.{regname}.precip.{varstr}{TimeStart[0:4]}-{TimeEnd[0:4]}.npy'
    if os.path.exists(index_fout) and OVERWRITE == True:
        os.remove(index_fout)
    if os.path.exists(preci_fout) and OVERWRITE == True:
        os.remove(preci_fout)

    np.save(index_fout, fullinds)
    np.save(preci_fout, WTprecip)

    CentroidsAct = np.reshape(CentroidsAct, (numWTs, LatWT.shape[0], LonWT.shape[0], nvar))
    CentroidsAct = np.moveaxis(CentroidsAct, 3, -3)

    print('|===| Saving centroids to file')
    centroids = xr.Dataset(
        data_vars = dict(
        centroids = (['WT','var','lat','lon'], CentroidsAct),
        frequency = (['WT'], WTfreq),
        WT_precip = (['WT','lat','lon'], WTprecip),
    ),
    coords = dict(
        WT = (['WT'], np.arange(numWTs)),
        var = (['var'], VariableList),
        lat = (['lat'], LatWT.data),
        lon = (['lon'], LonWT.data),
        ),
    )

    centroids.to_netcdf(diro+filo)
    
    filo = diro + 'norm_factors/' + regname + '_daily_vars_mean.pkl'
    with open(filo, 'wb') as file:
        pickle.dump(dvmean, file)
    filo = diro + 'norm_factors/' + regname + '_daily_vars_std.pkl'
    with open(filo, 'wb') as file:
        pickle.dump(dvstd, file)
    filo = diro + 'norm_factors/' + regname + '_rgrClustersFin.pkl'
    with open(filo, 'wb') as file:
        pickle.dump(rgrClustersFin, file)


def WT_lite(varsNorm, nlat, nlon, rdpr, rdinds, rdntim, TimeStart, TimeEnd, numberWTs=0):
    ## LIGHTER VERSION OF WT CODE TO JUST GET PRECIPITATION BY WT FOR OPTIMAL CLUSTER TEST

    rgrClustersFin=ClusterAnalysis(varsNorm, './', 0, TimeStart[0:4]+'-'+TimeEnd[0:4], Plot=0, numWT=numberWTs)
    CentroidsAct=rgrClustersFin[0]
    numWTs = CentroidsAct.shape[0]
    WTfreq = np.zeros(numWTs)

    WTprecip = np.zeros((numWTs, nlat, nlon))
    WT_index = {}
    fullinds = {}
    for i in range(numWTs):
        wt = str(i)
        WT_index[wt] = np.where(rgrClustersFin[1] == i)[0]
        tmp = WT_index[wt]
        WTfreq[i] = round((np.count_nonzero(rgrClustersFin[1] == i)/rdntim)*100, 1)
        WTprecip[i,:,:] = np.mean(rdpr[tmp,:,:], axis=0)
        fullinds[wt] = rdinds[WT_index[wt]]
    return WTprecip


def compute_wtpr_cmip5(regind, imod, TimeStart='1981-01-01', TimeEnd='2016-12-31', numWTs=6, OVERWRITE=True):
    timslic = slice(TimeStart,TimeEnd)
    regions, VariableList = get_regional_WT_varlist()
    models, methods, variants, lilmodels, _ = frdc5.get_ds_metavars(FULL_LIST=True)
    mod = models[imod]
    lilmod = lilmodels[imod]
    variant = variants[imod]
    reg = regions[regind]
    varstr = ''
    regname = reg.replace(' ', '')
    for var in VariableList[reg]:
        varstr += var+'.'

    _, _, centroids = load_WT_centroids(regname, varstr, TimeStart=TimeStart, TimeEnd=TimeEnd)

    WT_len = centroids['centroids'].shape[0]
    lonmin = centroids['lon'][0].data.item()
    lonmax = centroids['lon'][-1].data.item()
    lonslic = slice(lonmin, lonmax)
    latmin = centroids['lat'][0].data.item()
    latmax = centroids['lat'][-1].data.item()
    latslic = slice(latmin, latmax)

    diro = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/{regname}/'

    print('|===| Beginning processing for model: ' + mod)
    _, _, varsNorm, _, rdntim, rdinds = \
    create_norm_array(mod, TimeStart, TimeEnd, latslic, lonslic, reg, VariableList[reg])
    WT_index = categorize_precip_WT(regname, varsNorm, rdntim)

    # Compute raw GCM weather types:
    nci, _ = load_raw(mod)
    nci = nci[['pr']].sel(time=timslic, lat=latslic, lon=lonslic,drop=True)
    nci['pr'] = nci['pr']*86400
    ncc = {}
    for iwt in range(numWTs):
        ncc[str(iwt)] = nci.isel(time=rdinds[WT_index[str(iwt)]])
        filo = f'{regname}.{mod}.raw.WT{iwt}.{varstr}RDprecip_alltim.nc'
        if os.path.exists(diro+filo) and not OVERWRITE:
            continue
        else:
            ncc[str(iwt)].to_netcdf(f'{diro}{regname}.{mod}.raw.WT{iwt}.{varstr}RDprecip_alltim.nc')
    del nci

    for meth in methods:
        if os.path.exists(f'{diro}{regname}.{mod}.{meth}.WT5.{varstr}RDprecip_alltim.nc') and not OVERWRITE:
            continue
        if frdc5.check_modmeth(mod, meth, reg=reg):
            print('|===| Beginning processing for method: ' + meth)
        else:
            continue

        dspr = frdc5.read_ds_pr_cmip5(meth, mod, timslic, lilmod, variant, TAS=False)[['pr']]
        if dspr is None:
            continue
        if dspr['lon'][0] < 0:
            dspr['lon'] = dspr['lon'] + 360.
        print('|===| Beginning processing for region: ' + reg)

        for j in range(WT_len):
            wt = str(j)
            filo = f'{regname}.{mod}.{meth}.WT{wt}.{varstr}RDprecip_alltim.nc'
            if os.path.exists(diro+filo) and not OVERWRITE:
                continue
            
            print('|===|===|===| Beginning processing for WT: ' + wt)
            ncc_times_ymd = to_ymd(ncc[wt]['time'].values)
            dspr_times_ymd = to_ymd(dspr['time'].values)
            common_times = np.intersect1d(ncc_times_ymd, dspr_times_ymd)
            if len(common_times) < (len(ncc_times_ymd)*.95):
                print('|===|===|=== WARNING: NOT ENOUGH COMMON TIMES BETWEEN RAW AND ' + mod + ' ' + meth + ' FOR WT ' + str(wt))
                print('|===|===|=== Number of common times: ' + str(len(common_times)))
            times = ncc[wt]['time'].values[np.isin(ncc_times_ymd, common_times)]
            dspr.sel(lat=latslic, lon=lonslic).isel(time=np.isin(dspr_times_ymd, common_times)).assign_coords(time=times).to_netcdf(diro+filo)
            print('|===|===|===| Successfully saved WT precip for WT: ' + wt)


def compute_wtpr_obs(regind, obsind, TimeStart='1981-01-01', TimeEnd='2016-12-31', numWTs=6, OVERWRITE=True):
    timslic = slice(TimeStart,TimeEnd)
    obsdsetlist = ['ERA5', 'PRISM', 'CONUS404', 'Livneh', 'NLDAS', 'GMET', 'nClimGrid', 'gridMET']
    obs = obsdsetlist[obsind]
    regions, VariableList = get_regional_WT_varlist()
    reg = regions[regind]
    regname = reg.replace(' ', '')
    varstr = ''
    for var in VariableList[reg]:
        varstr += var+'.'
    
    WT_index_era5, _, centroids = load_WT_centroids(regname, varstr)

    WT_len = centroids['centroids'].shape[0]
    lonmin = centroids['lon'][0].data.item()
    lonmax = centroids['lon'][-1].data.item()
    lonslic = slice(lonmin, lonmax)
    latmin = centroids['lat'][0].data.item()
    latmax = centroids['lat'][-1].data.item()
    latslic = slice(latmin, latmax)

    diro = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/{regname}/'

    # WT_index_era5_noleap = {}
    # for i in range(numWTs):
    #     wt = str(i)
    #     times = pd.date_range(TimeStart, TimeEnd, freq='D')
    #     leap_day_mask = (times.month == 2) & (times.day == 29)
    #     leap_day_indices = np.where(leap_day_mask)[0]
    #     no_leap_inds = []
    #     for j in range(len(WT_index_era5[wt])):
    #         idx = int(WT_index_era5[wt][j])
    #         if idx in leap_day_indices:
    #             continue  # Skip leap days
    #         else:
    #             # Find how many leap days have occurred before or at this index
    #             shift = np.searchsorted(leap_day_indices, idx, side='right')
    #             no_leap_inds.append(idx - shift)
    #     WT_index_era5_noleap[wt] = np.array(no_leap_inds, dtype=int)

    # era5pr = fro.read_obs_daily('ERA5', timslic)['pr']
    # landsea_nat = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/OBS/ERA5/era5.landseamask.nc')
    # regr = xesmf.Regridder(landsea_nat, era5pr, method='nearest_s2d')
    # landsea = regr(landsea_nat)
    # era5_wtpr = xr.where(np.squeeze(landsea) <= 0.5, np.nan, era5pr).sel(lon=lonslic, lat=latslic)
    # era5_wtpr = remove_leap_days(era5_wtpr.rename({'lsm': 'pr'}))
    # era5_times_ymd = to_ymd(era5_wtpr['time'].values)
    # else:
    if obs == 'ERA5':
        obspr = fro.read_obs_daily(obs, timslic)[['pr']]
    else:
        obspr = fro.read_obs_daily(obs, timslic, ICARGRID=True)[['pr']]

    # obspr_times_ymd = to_ymd(obspr['time'].values)
    # common_times = np.intersect1d(era5_times_ymd, obspr_times_ymd)
    # if len(common_times) < (len(era5_wtpr['time'])):
    #     print('|===|===|=== WARNING: VERY FEW COMMON TIMES BETWEEN ERA5 AND ' + obs + ' FOR WT ' + str(wt))
    #     print('|===|===|=== Number of common times: ' + str(len(common_times)))
    # obspr = obspr.isel(time=np.isin(obspr_times_ymd, common_times))
    # times = era5_wtpr['time'].values[np.isin(era5_times_ymd, common_times)]
    # obspr = obspr.assign_coords(time=times)

    for j in range(WT_len):
        wt = str(j)
        filo = regname + '.' + obs + '.WT' + wt + '.' + varstr + 'RDprecip_alltim.nc'
        if os.path.exists(diro+filo) and not OVERWRITE:
            continue
        print('|===|===|===|===| Beginning processing for ' + obs + ' for WT: ' + wt)

        # if obs == 'ERA5':
        #     tmp = obspr.copy().isel(time=WT_index_era5[wt])
        #     # wtpr = xr.where(np.squeeze(landsea) == 0.0, np.nan, tmp).sel(lon=lonslic, lat=latslic)
        #     # wtpr = wtpr.rename({'lsm': 'pr'})
        # else:
        wtpr = obspr.copy().isel(time=WT_index_era5[wt]).sel(lon=lonslic, lat=latslic)
        wtpr.to_netcdf(diro+filo)
        del wtpr

def compute_wtpr_GARDLENS(regind, TimeStart='1981-01-01', TimeEnd='2016-12-31', numWTs=6, OVERWRITE=True):
    timslic = slice(TimeStart,TimeEnd)
    regions, VariableList = get_regional_WT_varlist()
    enslist = frdc5.CESM2_LE_ensemble_list()
    reg = regions[regind]

    OVERWRITE=True
    varstr = ''
    regname = reg.replace(' ', '')
    for var in VariableList[reg]:
        varstr += var+'.'

    _, _, centroids = load_WT_centroids(regname, varstr)

    lonmin = centroids['lon'][0].data.item()
    lonmax = centroids['lon'][-1].data.item()
    lonslic = slice(lonmin, lonmax)
    latmin = centroids['lat'][0].data.item()
    latmax = centroids['lat'][-1].data.item()
    latslic = slice(latmin, latmax)

    diro = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/{regname}/GARDLENS/'

    diri = '/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/'
    for ens in enslist:
        print('|===| Beginning processing for model: CESM2')
        _, _, varsNorm, _, rdntim, rdinds = \
        create_norm_array('CESM2-LE', TimeStart, TimeEnd, latslic, lonslic, reg, VariableList[reg], ens=ens)
        WT_index = categorize_precip_WT(regname, varsNorm, rdntim)

        # Compute raw GCM weather types:
        nci = xr.open_dataset(f'{diri}CESM2-LE_WTvars_CONUS_{ens}_1981-2016.hist_1deg.nc')[['pr','tas']].sel(time=timslic, lat=latslic, lon=lonslic, drop=True)
        if nci['lon'][0] < 0:
            nci['lon'] = nci['lon'] + 360.
        for iwt in range(6):
            print(WT_index[str(iwt)])
            nci.isel(time=rdinds[WT_index[str(iwt)]]).to_netcdf(f'{diro}{regname}.CESM2-LE.{ens}.raw.WT{iwt}.{varstr}RDprecip_alltim.nc')

        # Load GARDLENS data and select indices that match each WT determined from CESM2-LE members
        dspr = frdc5.read_ds_pr_cmip5('GARDLENS', 'cesm2', timslic, None, ens.replace('.0','_'), scenario='hist')
        if dspr['lon'][0] < 0:
            dspr['lon'] = dspr['lon'] + 360.
        print('|===| Beginning processing for region: ' + reg)

        for j in range(numWTs):
            wt = str(j)
            filo = f'{regname}.GARDLENS.CESM2-LE.{ens}.WT{wt}.{varstr}RDprecip_alltim.nc'
            if os.path.exists(diro+filo) and not OVERWRITE:
                continue
            print('|===|===|===| Beginning processing for WT: ' + wt)
            wtarr = rdinds[WT_index[wt]]
            wtpr = dspr.copy()
            wtpr.isel(time=wtarr).sel(lat=latslic, lon=lonslic).to_netcdf(diro+filo)
            del wtpr


def compute_rmse_map(regind, TimeStart = '1981-01-01', TimeEnd = '2016-12-31', numWTs = 6):
    """
    Computes RMSE and climatology maps for weather type (WT) precipitation across a specified region.
    This function processes observational and model datasets to calculate RMSE (Root Mean Square Error) and 
    climatology metrics for precipitation associated with different weather types. It performs regridding, 
    time matching, and metric calculations for each WT, observational dataset, model, and downscaling method. 
    The results are saved as NetCDF files containing metric maps for both model/method combinations and observational datasets.
    All parameters should match those used in the initial WT computation or it will fail or lead to incorrect results.
    
    Parameters
    ----------
    regind : int
        Index of the region to process.  Used to parallelize over multiple regions.
    TimeStart : str, optional
        Start date for the analysis period (default is '1981-01-01').
    TimeEnd : str, optional
        End date for the analysis period (default is '2016-12-31').
    numWTs : int, optional
        Number of weather types to process (default is 6).
    Saves
    -----
    NetCDF files containing RMSE and climatology maps for each WT, region, model, method, and observational dataset.
    """

    # Set up parameters
    regions, VariableList = get_regional_WT_varlist()
    diri = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/'
    models, methods, _, _, _ = frdc5.get_ds_metavars(FULL_LIST=True)
    obsdsetlist = ['PRISM', 'CONUS404', 'Livneh', 'NLDAS', 'GMET', 'nClimGrid', 'gridMET']

    reg = regions[regind]
    regname = reg.replace(' ', '')
    varstr = ''
    for var in VariableList[reg]:
        varstr += var+'.'

    # Suppress warnings because they whine about NaN points that are supposed to be there
    # and non C-continguous arrays
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(UserWarning)

        regname = reg.replace(' ', '')
        diri = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/'
        print('|=== Beginning processing for region: ' + reg)
        nmod = len(models)
        nmeth = len(methods)
        nobs = len(obsdsetlist)
        for wt in range(numWTs):
            # Prepare regridders by loading ERA5 and PRISM WT precip datasets
            print('|===|=== WT: ' + str(wt))
            tmp_wtpr = xr.open_dataset(f'{diri}{regname}/{regname}.PRISM.WT{wt}.{varstr}RDprecip_alltim.nc')
            wtpr_era5 = xr.open_dataset(f'{diri}{regname}/{regname}.ERA5.WT{wt}.{varstr}RDprecip_alltim.nc')
            wtpr_era5['mask'] = xr.where(np.isnan(wtpr_era5['pr'].isel(time=0)), 0, 1)
            tmp_wtpr['mask'] = xr.where(np.isnan(tmp_wtpr['pr'].isel(time=0)), 0, 1)
            regr_obs = xesmf.Regridder(wtpr_era5, tmp_wtpr, method='nearest_s2d')
            era5_wtpr_ds = regr_obs(wtpr_era5['pr']).to_dataset(name='pr')
            del wtpr_era5

            # Initialize metric arrays on first loop
            if wt == 0:
                lon = tmp_wtpr['lon']
                lat = tmp_wtpr['lat']
                nlon = len(lon)
                nlat = len(lat)
                wtpr_clim_obs = np.full((numWTs, nobs, nlat, nlon), np.nan, dtype=float)
                wtpr_rmse_obs = np.full((numWTs, nobs, nlat, nlon), np.nan, dtype=float)
                wtpr_clim_ds  = np.full((numWTs, nmod, nmeth, nlat, nlon), np.nan, dtype=float)
                wtpr_rmse_ds  = np.full((numWTs, nmod, nmeth, nlat, nlon), np.nan, dtype=float)

            del tmp_wtpr

            # Compute WT metrics for each observational dataset
            # Here RMSE maps are computed against ERA5 regridded to 1 degree,
            # then nearest neighbor regridded to the 1/8th degree grid
            for obsind, obs in enumerate(obsdsetlist):
                tmpobs = xr.open_dataset(f'{diri}{regname}/{regname}.{obs}.WT{wt}.{varstr}RDprecip_alltim.nc')
                print('Length of ERA5 WT PR time: ' + str(len(era5_wtpr_ds['time'])))
                print('Length of ' + obs + ' time: ' + str(len(tmpobs['time'])))

                ## Calendar handling is remarkably sensitive, but converting each entry to strings
                ## to find common dates seems to work best.  They should match for almost all dates,
                ## barring a few at the edges or leap days
                # era5_times_ymd = to_ymd(era5_wtpr_ds['time'].values)
                # tmpobs_times_ymd = to_ymd(tmpobs['time'].values)
                # common_times = np.intersect1d(era5_times_ymd, tmpobs_times_ymd)
                # if len(common_times) < (len(era5_wtpr_ds['time'])*.90):
                #     print('|===|===|=== WARNING: NOT ENOUGH COMMON TIMES BETWEEN ERA5 AND ' + obs + ' FOR WT ' + str(wt))
                #     print('|===|===|=== Number of common times: ' + str(len(common_times)))
                # tmpobs = tmpobs.isel(time=np.isin(tmpobs_times_ymd, common_times))
                # times = era5_wtpr_ds['time'].values[np.isin(era5_times_ymd, common_times)]
                # tmpobs = tmpobs.assign_coords(time=times)
                # tmpera5 = era5_wtpr_ds.isel(time=np.isin(era5_times_ymd, common_times))


                # wtpr_rmse_obs[wt, obsind, :, :] = xs.rmse(tmpobs['pr'], tmpera5['pr'], dim='time', skipna=True).data
                try:
                    wtpr_rmse_obs[wt, obsind, :, :] = xs.rmse(tmpobs['pr'], era5_wtpr_ds['pr'], dim='time', skipna=True).data
                except xr.structure.alignment.AlignmentError:
                    print('|===|===|=== WARNING: ALIGNMENT ERROR BETWEEN ERA5 AND ' + obs + ' FOR WT ' + str(wt))
                    tmpobs['time'] = era5_wtpr_ds['time']
                    wtpr_rmse_obs[wt, obsind, :, :] = xs.rmse(tmpobs['pr'], era5_wtpr_ds['pr'], dim='time', skipna=True).data
                wtpr_clim_obs[wt, obsind, :, :] = tmpobs['pr'].mean(dim='time').data
                del tmpobs
                # del tmpera5

            # Compute WT metrics for each model/method combination.
            # Here RMSE maps are computed against RAW model data, first conservatively regridded to 1 degree,
            # then nearest neighbor regridded to the 1/8th degree grid, as for ERA5 above.
            for imod, mod in enumerate(models):
                print('|===|===|=== Model: ' + mod)

                raw_wtpr_og = xr.open_dataset(f'{diri}{regname}/{regname}.{mod}.raw.WT{wt}.{varstr}RDprecip_alltim.nc')
                raw_wtpr_og['mask'] = xr.where(np.isnan(raw_wtpr_og['pr'].isel(time=0)), 0, 1)
                icar = xr.open_dataset(f'{diri}{regname}/{regname}.{mod}.ICAR.WT{wt}.{varstr}RDprecip_alltim.nc')
                icar['mask'] = xr.where(np.isnan(icar['pr'].isel(time=0)), 0, 1)
                raw_regr = xesmf.Regridder(raw_wtpr_og, icar, method='nearest_s2d')
                raw_wtpr_ds = raw_regr(raw_wtpr_og['pr']).to_dataset(name='pr')
                del icar

                for imeth, meth in enumerate(methods):
                    if not frdc5.check_modmeth(mod, meth, reg=reg):
                        continue
                    print('|===|===|=== Method: ' + meth)

                    wtpr = xr.open_dataset(f'{diri}{regname}/{regname}.{mod}.{meth}.WT{wt}.{varstr}RDprecip_alltim.nc')

                    # Match times between raw and downscaled datasets, as for observations above
                    raw_times_ymd = to_ymd(raw_wtpr_ds['time'].values)
                    wtpr_times_ymd = to_ymd(wtpr['time'].values)
                    common_times = np.intersect1d(raw_times_ymd, wtpr_times_ymd)
                    if len(common_times) < (len(raw_wtpr_ds['time'])*.90):
                        print('|===|===|=== WARNING: NOT ENOUGH COMMON TIMES BETWEEN RAW AND ' + mod + ' ' + meth + ' FOR WT ' + str(wt))
                        print('|===|===|=== Number of common times: ' + str(len(common_times)))
                    wtpr = wtpr.isel(time=np.isin(wtpr_times_ymd, common_times))
                    times = raw_wtpr_ds['time'].values[np.isin(raw_times_ymd, common_times)]
                    wtpr = wtpr.assign_coords(time=times)
                    tmpraw = raw_wtpr_ds.isel(time=np.isin(raw_times_ymd, common_times))

                    wtpr_rmse_ds[wt, imod, imeth, :, :] = xs.rmse(wtpr['pr'], tmpraw['pr'], dim='time', skipna=True).data
                    wtpr_clim_ds[wt, imod, imeth, :, :] = wtpr['pr'].mean(dim='time').data
                    del wtpr
                    del tmpraw
    
    # Save metric maps to file
    ds_metrics = xr.Dataset(
        data_vars={
        'wtpr_clim_ds': (('WT', 'model', 'method', 'lat', 'lon'), wtpr_clim_ds),
        'wtpr_rmse_ds': (('WT', 'model', 'method', 'lat', 'lon'), wtpr_rmse_ds),
        },        
        coords={
            'WT': np.arange(numWTs),
            'model': models,
            'method': methods,
            'lon': lon,
            'lat': lat,
        }
    )
    obs_metrics = xr.Dataset(
        data_vars={
        'wtpr_clim_obs': (('WT', 'obs', 'lat', 'lon'), wtpr_clim_obs),
        'wtpr_rmse_obs': (('WT', 'obs', 'lat', 'lon'), wtpr_rmse_obs),
        },        
        coords={
            'WT': np.arange(numWTs),
            'obs': obsdsetlist,
            'lon': lon,
            'lat': lat,
        }
    )
    diro = f'{diri}wtpr_metric_maps/'
    ds_filo = f'{regname}_wtpr_DS_metric_maps_{TimeStart[0:4]}-{TimeEnd[0:4]}.nc'
    ds_metrics.to_netcdf(f'{diro}{ds_filo}')
    obs_filo = f'{regname}_wtpr_obs_metric_maps_{TimeStart[0:4]}-{TimeEnd[0:4]}.nc'
    obs_metrics.to_netcdf(f'{diro}{obs_filo}')


def to_ymd(arr):
    """
    Converts an array of date-like objects to normalized pandas Timestamps (YYYY-MM-DD).
    Parameters
    ----------
    arr : array-like
        Array of date objects (e.g., np.datetime64, cftime.datetime).
    Returns
    -------
    pandas.Series or pandas.DatetimeIndex
        Array of normalized pandas Timestamps.
    """

    if isinstance(arr[0], np.datetime64):
        return pd.to_datetime(arr).normalize()
    else:
        # cftime.datetime or similar
        return pd.to_datetime([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in arr])
                                

def compute_rmse_map_GARDLENS(regind, numWTs = 6):
    """
    Computes RMSE maps and climatologies for weather type precipitation (WTpr) using the GARDLENS method
    for a specified region and ensemble members.
    Parameters
    ----------
    regind : int
        Index of the region to process.
    numWTs : int, optional
        Number of weather types to process (default is 6).
    Returns
    -------
    None
        Saves an xarray Dataset containing RMSE maps and climatologies to NetCDF file.
    Notes
    -----
    - This function is tailored for the GARDLENS method and CESM2-LE ensemble.
    - Output includes RMSE maps, WTpr climatology (GARDLENS), and WTpr climatology (raw).
    - Results are saved to a NetCDF file in the specified output directory.
    """

    regions, VariableList = get_regional_WT_varlist()
    reg = regions[regind]
    varstr = ''
    regname = reg.replace(' ', '')
    for var in VariableList[reg]:
        varstr += var+'.'
    diri = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/{regname}/GARDLENS/'
    enslist = frdc5.CESM2_LE_ensemble_list()

    tester = xr.open_dataset(f'{diri}{regname}.GARDLENS.CESM2-LE.{enslist[0]}.WT0.{varstr}RDprecip_alltim.nc').isel(time=0)
    lon = tester['lon']
    lat = tester['lat']
    nlon = len(tester['lon'])
    nlat = len(tester['lat'])
    del tester
    rmse_map_ens = np.full((numWTs, len(enslist), nlat, nlon), np.nan, dtype=float)
    wtpr_clim_ens = np.full((numWTs, len(enslist), nlat, nlon), np.nan, dtype=float)
    wtpr_clim_raw = np.full((numWTs, len(enslist), nlat, nlon), np.nan, dtype=float)

    for iens, ens in enumerate(enslist):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn(UserWarning)
            print('|=== Beginning processing for region: ' + reg)

            for wt in range(numWTs):
                print('|===|=== WT: ' + str(wt))

                raw_wtpr = xr.open_dataset(f'{diri}{regname}.CESM2-LE.{ens}.raw.WT{wt}.{varstr}RDprecip_alltim.nc')
                wtpr = xr.open_dataset(f'{diri}{regname}.GARDLENS.CESM2-LE.{ens}.WT{wt}.{varstr}RDprecip_alltim.nc')
                wtpr['time'] = raw_wtpr['time']
                wtpr['mask'] = xr.where(np.isnan(wtpr['pr'].isel(time=0)), 0, 1)
                regr = xesmf.Regridder(raw_wtpr, wtpr, method='nearest_s2d')
                raw_wtpr_ds = regr(raw_wtpr)

                rmse_map_ens[wt, iens, :, :]  = xs.rmse(wtpr['pr'], raw_wtpr_ds['pr'], dim='time', skipna=True).data
                wtpr_clim_ens[wt, iens, :, :] = wtpr['pr'].copy().mean(dim='time').data
                wtpr_clim_raw[wt, iens, :, :] = raw_wtpr_ds['pr'].copy().mean(dim='time').data

                del wtpr
                del raw_wtpr_ds

    metrics = xr.Dataset(
        data_vars={
            'wtpr_rmse_ds': (('WT', 'ens', 'lat', 'lon'), rmse_map_ens),
            'wtpr_clim_ds': (('WT', 'ens', 'lat', 'lon'), wtpr_clim_ens),
            'wtpr_clim_raw': (('WT', 'ens', 'lat', 'lon'), wtpr_clim_raw),
        },        
        coords={
            'WT': np.arange(numWTs),
            'ens': enslist,
            'lon': lon,
            'lat': lat,
        }
    )
    diro = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/wtpr_metric_maps/GARDLENS/'
    filo = f'{regname}_wtpr_metric_maps_GARDLENS_CESM2-LE.nc'
    metrics.to_netcdf(f'{diro}{filo}')


def Optimal_Cluster_Test(regind, numWTs=6, TimeStart='1981-01-01', TimeEnd='2016-12-31'):
    """
    Runs an optimal cluster test for weather typing over a region.

    - Generates all variable combinations (size 2-5) from a set list.
    - Normalizes data and evaluates mean variance of WT precipitation clusters for each combination.
    - Saves results incrementally for restart and parallelization.

    Output: Pickle files with variance results for each combination.
    """

    regions, buko = fdse.make_buko_region_map(res='1deg', COMPUTE=False)
    # return regions, bukods
    region = regions[regind]
    reg = region
    regname = region.replace(' ', '')

    diri = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/varcomp_outputs/'
    diro = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/varcomp_outputs/'

    fcomplete = sorted(glob.glob(f'{diro}{regname}.*.VARI_varcomp.pkl'))
    if len(fcomplete) == 33:
        print('All output files found for region: ' + regname)
        quit()
    elif len(fcomplete) >= 2:
        frestart = fcomplete[-1]
        irestart = int(frestart.split('.')[-3])
    elif len(fcomplete) == 1:
        frestart = fcomplete[0]
        irestart = int(frestart.split('.')[-3])
    elif len(fcomplete) == 0:
        irestart = 0

    VariableList = [
        'U850', 'U500', 'U250',
        'V850', 'V500', 'V250',
        'UV850', 'UV500', 'UV250',
        'Q850', 'Q500', 'ZG500', 'MFL500',
        'PW', 'PSL'
    ]

    combinations_list = []
    for r in range(2, 6):
        combinations_list.extend(list(comb) for comb in combinations(VariableList, r))
    ncombos = len(combinations_list)

    if reg == regions[0]:
        filo = f'{diro}varcombos.pkl'
        with open(filo, 'wb') as file:
            pickle.dump(combinations_list, file)

    varinds = {}
    lstind = 0
    for lstind, lst in enumerate(combinations_list):
        varinds[lstind] = np.zeros(len(lst), dtype=int)
        for i, var in enumerate(lst):
            varinds[lstind][i] = int(VariableList.index(var))

    lonmin, lonmax, latmin, latmax = fdse.get_region_edges(regions, buko)
    latrng = slice(latmin[reg], latmax[reg])
    lonrng = slice(lonmin[reg], lonmax[reg])
    _, _, varsNorm, rdpr, rdntim, rdinds = create_norm_array('era5', TimeStart, TimeEnd, latrng, lonrng, regname, VariableList)
    LonWT = rdpr['lon']
    nlon = len(LonWT)
    LatWT = rdpr['lat']
    nlat = len(LatWT)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)

        print('Running cluster analysis')

        if irestart == 0:
            test_vari = np.full((ncombos), np.nan)
        else:
            if irestart < 10000:
                fili = f'{diri}{regname}.0{irestart}.VARI_varcomp.pkl'
            else:
                fili = f'{diri}{regname}.{irestart}.VARI_varcomp.pkl'
            with open(fili, 'rb') as file:
                test_vari = pickle.load(file)

        for icomb in range(irestart, ncombos):
            varsNormTest = varsNorm[:, :, :, varinds[icomb]]
            mWTprecip = WT_lite(varsNormTest, nlat, nlon, rdpr, rdinds, rdntim, TimeStart, TimeEnd, numberWTs=numWTs)

            test_vari[icomb] = 0.0
            wtstd = np.var(mWTprecip, axis=0)
            test_vari[icomb] = np.nanmean(wtstd)


            if icomb != irestart:
                # if (icomb % 1000 == 0) or (icomb == ncombos - 1):
                if (icomb == ncombos - 1):
                    if icomb < 10000:
                        filo = f'{diro}{regname}.0{icomb}.VARI_varcomp.pkl'
                    else:
                        filo = f'{diro}{regname}.{icomb}.VARI_varcomp.pkl'
                    with open(filo, 'wb') as file:
                        pickle.dump(test_vari, file)