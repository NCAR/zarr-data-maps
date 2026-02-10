import glob
import numpy as np
import os
import xarray as xr
import xskillscore as xs
import funcs_general as fg
import xesmf
import funcs_read_ds_cmip5 as frdc5
import funcs_read_obs as fro
import pandas as pd
import warnings
from scipy.stats import genextreme
from scipy.ndimage import generic_filter
# get_ds_metavars, check_modmeth

def make_buko_region_map(res, COMPUTE=False):
    diri = '/glade/work/nlybarger/region-masks/'
    fili = {}
    fili['North Atlantic'] = ['NorthAtlantic']
    fili['Mid Atlantic'] = ['Appalachia', 'MidAtlantic']
    fili['Gulf Coast'] = ['SPlains', 'DeepSouth', 'Southeast', 'Mezquital']
    fili['Pacific Northwest'] = ['PacificNW']
    fili['Pacific Southwest'] = ['PacificSW']
    fili['Northern Plains'] = ['NPlains', 'CPlains', 'Prairie']
    fili['Mountain West'] = ['NRockies', 'SRockies', 'GreatBasin']
    fili['Great Lakes'] = ['GreatLakes', 'EastBoreal']
    fili['Desert Southwest'] = ['Southwest', 'Mezquital']
    regions = list(fili.keys())
    if COMPUTE:
        regions = list(fili.keys())
        mask = {}
        for reg in regions:
            j = 0
            for i in fili[reg]:
                if j == 0:
                    mask[reg] = xr.open_dataset(diri+i+'.nc', engine='netcdf4')
                    j = 1
                else:
                    tmp = xr.open_dataset(diri+i+'.nc', engine='netcdf4')
                    mask[reg] = xr.where(tmp == 1.0, tmp, mask[reg])
            if reg == 'Gulf Coast':
                mask[reg] = mask[reg].where(mask[reg]['lon'] >= 256.5, np.nan)
            elif reg == 'Desert Southwest':
                mask[reg] = mask[reg].where(mask[reg]['lon'] < 256.5, np.nan)
        for i in range(len(regions)):
            reg = regions[i]
            if i == 0:
                nlon = len(mask[reg]['lon'])
                nlat = len(mask[reg]['lat'])
                buko = np.zeros((nlat,nlon))
            buko[mask[reg]['mask']==1.0] = i+1
        buko[buko == 0.] = np.nan
        bukods = xr.Dataset(
            data_vars=dict(
                mask=(['lat','lon'], buko),
            ),
            coords=dict(
                lat=mask[reg]['lat'].data,
                lon=mask[reg]['lon'].data,
            ),
        )
        bukods = xr.where(bukods['lat'] <= 49.5, bukods, np.nan)
        bukods = bukods.sel(lat=slice(23, 51), lon=slice(230, 301), drop=True)


        gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
        bukods = bukods.rename({'mask': 'regions'})
        bukods['mask'] = xr.where(np.isnan(bukods['regions']), np.nan, 1.)
        regr = xesmf.Regridder(bukods, gridtar, 'nearest_s2d')
        targ = regr(bukods)
        targ = targ.where(~np.isnan(gridtar['mask']))
        # Select indices for longitude and latitude ranges


        ## Making some custom changes to Great Lakes and Desert Southwest that are appropriate because of the strict CONUS borders
        # Add some areas around the Great Lakes to that region, especially around Ohio and New York 
        lon_mask = (targ.lon >= 276) & (targ.lon <= 285)
        lat_mask = (targ.lat >= 40) & (targ.lat <= 45)

        # Get the mask array
        mask_arr = targ['regions'].values

        # Find nan locations
        nan_mask = np.isnan(mask_arr)
        region_mask = np.zeros_like(mask_arr, dtype=bool)
        for i, lat_val in enumerate(targ.lat.values):
            if not lat_mask[i]:
                continue
            for j, lon_val in enumerate(targ.lon.values):
                if not lon_mask[j]:
                    continue
                # Check if within 8 pixels of a Great Lake (only checking to the left and
                # above since otherwise we catch some pixels within 8 pixels of the Atlantic Ocean)
                i_min = i
                i_max = min(i + 8, mask_arr.shape[0])
                j_min = max(j - 8, 0)
                j_max = j
                if np.any(nan_mask[i_min:i_max, j_min:j_max]):
                    region_mask[i, j] = True
        targ['regions'].values[region_mask] = 8


        # Add some areas in the Desert Southwest to that region, especially the very Southern Rockies
        lon_mask_9 = (targ.lon >= 251) & (targ.lon <= 255)
        lat_mask_9 = (targ.lat >= 32) & (targ.lat <= 34)

        for i, lat_val in enumerate(targ.lat.values):
            if not lat_mask_9[i]:
                continue
            for j, lon_val in enumerate(targ.lon.values):
                if lon_mask_9[j]:
                    targ['regions'].values[i, j] = 9
        targ = targ.where(~np.isnan(gridtar['mask']))


        # Apply mode filter to smooth borders, but only on valid region values (1-9)
        mask_vals = targ['regions'].values.copy()
        smoothed_mask = np.where(np.isnan(mask_vals), np.nan, mode_filter(mask_vals, size=3))
        targ['regions'].values[:] = smoothed_mask

        dummygrid = xr.Dataset(
            data_vars = dict(
            ),
            coords = dict(
                lon = (['lon'], np.arange(230,300+1,1, dtype=np.float32)),
                lat = (['lat'], np.arange(20,52+1,1, dtype=np.float32)),
            ),
        )

        regr = xesmf.Regridder(targ, dummygrid, 'nearest_s2d')
        targ.to_netcdf('/glade/campaign/ral/hap/nlybarger/bukods_icargrid.nc')
        regr(targ).to_netcdf('/glade/campaign/ral/hap/nlybarger/bukods_1deg.nc')
    else:
        # Load the precomputed dataset
        if res == '1deg':
            bukods = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/bukods_1deg.nc')
        elif res == 'icargrid':
            bukods = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/bukods_icargrid.nc')
        else:
            raise ValueError("Invalid resolution specified. Use '1deg' or 'icargrid'.")
        return regions, bukods


def mode_filter(arr, size=3):
    def mode_func(x):
        vals = x[~np.isnan(x)]
        if len(vals) == 0:
            return np.nan
        counts = np.bincount(vals.astype(int))
        return np.argmax(counts)
    return generic_filter(arr, mode_func, size=size, mode='nearest')


def get_region_edges(regions, bukods):
    lonmin = {}
    lonmax = {}
    latmin = {}
    latmax = {}
    for regind, reg in enumerate(regions):
        lonmin[reg] = bukods['lon'].where(bukods['regions'] == regind+1).min().data.item()
        lonmax[reg] = bukods['lon'].where(bukods['regions'] == regind+1).max().data.item()
        latmin[reg] = bukods['lat'].where(bukods['regions'] == regind+1).min().data.item()
        latmax[reg] = bukods['lat'].where(bukods['regions'] == regind+1).max().data.item()
    return lonmin, lonmax, latmin, latmax


def Compute_CONUS_Metric_Maps(timslic, mod=None, meth=None, CMIP5=False, obsdset=None, OBS=False, ens=None, GARDLENS=False, CESM2_RAW=False, scenario='hist', OVERWRITE=False):
    ## Finished for CMIP5 models and methods, obs datasets, and GARDLENS ens
    ## Still need to complete for CESM2-LE raw
    TimeStart = timslic.start
    TimeEnd = timslic.stop
    datstr = TimeStart[0:4] + '-' + TimeEnd[0:4]
    
    if mod is not None and meth is not None:
        CMIP5=True
        models, _, variants, lilmods, cvdp_variants = frdc5.get_ds_metavars(FULL_LIST=True)
        imod = models.index(mod)
        vari = variants[imod]
        lilmod = lilmods[imod]
        cvdp_variant = cvdp_variants[imod]

        print('Beginning computation of metrics for:')
        print(scenario)
        print('===| Method: ' + meth)

        # Checks for valid model/method combination
        if not frdc5.check_modmeth(mod, meth):
            return

        diro = '/glade/work/nlybarger/downscaling_metrics/cmip5/'
        filo = f'{mod}.{meth}.{scenario}.{datstr}.ds.conus.metric.maps.nc'
        if os.path.exists(diro+filo) and OVERWRITE:
            os.remove(diro+filo)
        elif os.path.exists(diro+filo) and not OVERWRITE:
            print('Metrics already computed for: ' + meth + ' ' + mod)
            return
        print('===|===| Model: ' + mod)
        outstr = f'{mod}-{meth}'

    elif obsdset is not None:
        OBS=True
        diro = '/glade/work/nlybarger/downscaling_metrics/obs/'
        filo = f'{obsdset}.{datstr}.ds.conus.metric.maps.nc'
        outstr = obsdset
        if os.path.exists(diro+filo) and OVERWRITE:
            os.remove(diro+filo)
        elif os.path.exists(diro+filo) and not OVERWRITE:
            print('Metrics already computed for obs: ' + obsdset)
            return

    elif (ens is not None) and GARDLENS:
        diro = '/glade/work/nlybarger/downscaling_metrics/GARDLENS/'
        filo = f'GARDLENS.{ens}.{scenario}.{datstr}.ds.conus.metric.maps.nc'
        outstr = f'GARDLENS-{ens}'
        if os.path.exists(diro+filo) and OVERWRITE:
            os.remove(diro+filo)
        elif os.path.exists(diro+filo) and not OVERWRITE:
            print('Metrics already computed for: ' + ens + ' ' + 'GARDLENS')
            return

    elif (ens is not None) and not GARDLENS:
        CESM2_RAW=True
        diro = '/glade/work/nlybarger/downscaling_metrics/CESM2-LE/'
        filo = f'CESM2-LE.{ens}.{scenario}.{datstr}.ds.conus.metric.maps.nc'
        outstr = f'CESM2-LE-{ens}'
        if os.path.exists(diro+filo) and OVERWRITE:
            os.remove(diro+filo)
        elif os.path.exists(diro+filo) and not OVERWRITE:
            print('Metrics already computed for: ' + ens + ' ' + 'CESM2-LE')
            return
    else:
        print('Error: Must provide either mod/meth, obsdset, or ens')
        return

    ## =========================================================================================================
            ## Read and prep CVDP data
    ## =========================================================================================================
    if CMIP5:
        if scenario == 'hist':
            # scenstr = 'historical'
            # datestr = '1900-2005'
            cvdp_fili1 = f'/glade/work/nlybarger/data/hydromet/cmip5_cvdp_data/historical/{mod}{cvdp_variant}.cvdp_data.1900-2005.nc'
            cvdp1 = xr.open_dataset(cvdp_fili1, decode_times=False).load()[['nino34', 'pdo_timeseries_mon', 'amo_timeseries_mon']]
            cvdp1['time'] = pd.date_range('1900-01-01','2005-12-31', freq='1MS')
            cvdp_fili2 = f'/glade/work/nlybarger/data/hydromet/cmip5_cvdp_data/rcp45/{mod}{cvdp_variant}.cvdp_data.2006-2100.nc'
            cvdp2 = xr.open_dataset(cvdp_fili2, decode_times=False).load()[['nino34', 'pdo_timeseries_mon', 'amo_timeseries_mon']]
            cvdp2['time'] = pd.date_range('2006-01-01','2100-12-31', freq='1MS')
            cvdp = xr.concat([cvdp1, cvdp2], dim='time').sel(time=timslic)
            del cvdp1
            del cvdp2
        else:
            scenstr = scenario
            datestr = '2006-2100'
            cvdp_fili = f'/glade/work/nlybarger/data/hydromet/cmip5_cvdp_data/{scenstr}/{mod}{cvdp_variant}.cvdp_data.{datestr}.nc'
            cvdp = xr.open_dataset(cvdp_fili, decode_times=False).load()[['nino34', 'pdo_timeseries_mon', 'amo_timeseries_mon']]
            cvdp['time'] = pd.date_range('2006-01-01','2100-12-31', freq='1MS')
        cvdp=cvdp.sel(time=timslic)
        nci = frdc5.read_ds_pr_cmip5(meth, mod, timslic, lilmod, vari, scenario=scenario, TAS=True)
    elif OBS:
        indy = fro.read_clim_inds(timslic)
        nci = fro.read_obs_daily(obsdset, timslic, ICARGRID=True)
    elif GARDLENS:
        nci = frdc5.read_ds_pr_cmip5('GARDLENS', 'cesm2', timslic, None, ens, scenario='hist')
        tmpens = ens.replace('_', '.0')
        n34 = xr.open_dataset(f'/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/CESM2-LE_nino34_{tmpens}_1981-2016.nc')
        n34 = n34.sel(time=timslic)
    elif CESM2_RAW:
        nci = xr.open_dataset(f'/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/CESM2-LE_tas_pr_CONUS_{ens}_1981-2016.hist_1deg.nc')[['tas', 'tasmin', 'tasmax', 'pr']]
        nci = nci.sel(time=timslic)
        n34 = xr.open_dataset(f'/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/CESM2-LE_nino34_{ens}_1981-2016.nc')


    ## =========================================================================================================
            ## Read and prep downscaled data
    ## =========================================================================================================

    nci_mon = nci.resample(time='MS').mean()
    month_length = nci_mon.time.dt.days_in_month
    nci_mon['pr'] = nci_mon['pr']*month_length
    if CMIP5:
        nci_mon['n34'] = (['time'],cvdp['nino34'].data)
    elif OBS:
        nci_mon['n34'] = (['time'],indy['n34'].data)
    elif GARDLENS or CESM2_RAW:
        nci_mon['n34'] = (['time'],n34['n34'].data)

    ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ### MONTHLY METRICS ========================================================================================
    ## =========================================================================================================
            ## Compute correlation between monthly DJF Nino3.4-precipitation correlation maps
    ## =========================================================================================================
    seasavg = fg.seasonal_avg_vars(nci_mon, 'lat', 'lon')
    n34pr = xs.pearson_r(seasavg['DJF']['pranom'],  seasavg['DJF']['n34'],dim='time', skipna=True)
    n34t  = xs.pearson_r(seasavg['DJF']['tasanom'], seasavg['DJF']['n34'],dim='time', skipna=True)

    tpcorr = xs.pearson_r(seasavg['ANN']['pr'], seasavg['ANN']['tas'], dim='time', skipna=True)

    ## =========================================================================================================
    ## Compute 1-year, 2-year, and 5-year SPI
    ## =========================================================================================================
    spi_1yr = fg.compute_spi_xarray(nci_mon['pr'], 12)
    spi_2yr = fg.compute_spi_xarray(nci_mon['pr'], 24)
    spi_5yr = fg.compute_spi_xarray(nci_mon['pr'], 60)

    threshold = -1.5
    drought_1yr = (spi_1yr < threshold).sum(dim='time').astype('float')
    drought_2yr = (spi_2yr < threshold).sum(dim='time').astype('float')
    drought_5yr = (spi_5yr < threshold).sum(dim='time').astype('float')

    ## =========================================================================================================
            ## Compute P & T trend maps
    ## =========================================================================================================
    ptrend = fg.compute_trend(nci_mon,'pr')
    ttrend = fg.compute_trend(nci_mon,'tas')

    ## =========================================================================================================
        ## Total precipitation on days with temperatue < 1C
    ## =========================================================================================================
    snowaccum = nci['pr'].where(nci['tas'] < 1).resample(time='YS').sum(dim='time')

    ## =========================================================================================================
        ## Fraction of days with precipitation > 0
    ## =========================================================================================================
    wet_day_frac = (nci['pr'] > 0.).sum(dim='time') / nci['pr'].count(dim='time')

    ## =========================================================================================================
        ## Compute freeze-thaw days as days when tasmin < 0 and tasmax > 0
    ## =========================================================================================================
    if ('tasmin' in list(nci.variables)) and ('tasmax' in list(nci.variables)):
        # Compute annual average number of days with tasmin < 0 and tasmax > 0
        freezethaw = ((nci['tasmin'] < 0) & (nci['tasmax'] > 0)).resample(time='YS').sum(dim='time')
        freezethaw = freezethaw.mean(dim='time')
    elif 't_range' in list(nci.variables):
        # Compute annual average number of days with tasmin < 0 and tasmax > 0
        nci['tasmax'] = nci['tas'] + nci['t_range']/2
        nci['tasmin'] = nci['tas'] - nci['t_range']/2
        freezethaw = ((nci['tasmin'] < 0) & (nci['tasmax'] > 0)).resample(time='YS').sum(dim='time')
        freezethaw = freezethaw.mean(dim='time')

    ## =========================================================================================================
        ## GEV 100-, 50-, and 20-year return levels
    ## =========================================================================================================
    # Compute annual maxima
    annual_max = nci['pr'].resample(time='YS', skipna=True).max(dim='time', skipna=True)

    # Prepare array to store return levels
    gev_100yr = np.full(annual_max.shape[1:], np.nan)
    gev_50yr = np.full(annual_max.shape[1:], np.nan)
    gev_20yr = np.full(annual_max.shape[1:], np.nan)

    # Fit GEV and compute 50- and 20-year return level for each grid point
    for i in range(annual_max.shape[1]):  # lat
        for j in range(annual_max.shape[2]):  # lon
            data = annual_max[:, i, j].values
            if np.all(np.isnan(data)):
                continue
            data = data[~np.isnan(data)]
            data = data[np.isfinite(data)]

            c, loc, scale = genextreme.fit(data)
            gev_100yr[i, j] = genextreme.ppf(1 - 1/100, c, loc=loc, scale=scale)
            gev_50yr[i, j] = genextreme.ppf(1 - 1/50, c, loc=loc, scale=scale)
            gev_20yr[i, j] = genextreme.ppf(1 - 1/20, c, loc=loc, scale=scale)

    # Set extremely large values to NaN (e.g., above 1e5 mm)
    gev_100yr[gev_100yr > 1800.] = np.nan
    gev_50yr[gev_50yr > 1800.] = np.nan
    gev_20yr[gev_20yr > 1800.] = np.nan

    ## =========================================================================================================
            ## Precipitation/temperature extremes maps (90th, 99th)
    ## =========================================================================================================
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pr90 = nci['pr'].quantile(q=.9,dim=['time'])
        pr99 = nci['pr'].quantile(q=.99,dim=['time'])
        t90  = nci['tas'].quantile(q=.9,dim=['time'])
        t99  = nci['tas'].quantile(q=.99,dim=['time'])
        # if not CESM2_RAW:
        outdims = ['lat','lon']
        outcoords = dict(
            lat = (['lat'], nci['lat'].data),
            lon = (['lon'], nci['lon'].data),
        )
        # else:
        #     outdims = ['ens','lat','lon']
        #     outcoords = dict(
        #         ens = (['ens'], nci['ens'].data),
        #         lat = (['lat'], nci['lat'].data),
        #         lon = (['lon'], nci['lon'].data),
        #     )
        metrics = xr.Dataset(
            data_vars = dict(
                n34pr        = xr.DataArray(n34pr.data, dims=outdims, attrs={'units':'unitless'}),
                n34t         = xr.DataArray(n34t.data, dims=outdims, attrs={'units':'unitless'}),
                ptrend       = xr.DataArray(ptrend.data, dims=outdims, attrs={'units':'mm/decade'}),
                ttrend       = xr.DataArray(ttrend.data, dims=outdims, attrs={'units':'°C/decade'}),
                pr90         = xr.DataArray(pr90.data, dims=outdims, attrs={'units':'mm'}),
                pr99         = xr.DataArray(pr99.data, dims=outdims, attrs={'units':'mm'}),
                pr_gev_20yr  = xr.DataArray(gev_20yr.data, dims=outdims, attrs={'units':'mm'}),
                pr_gev_50yr  = xr.DataArray(gev_50yr.data, dims=outdims, attrs={'units':'mm'}),
                pr_gev_100yr = xr.DataArray(gev_100yr.data, dims=outdims, attrs={'units':'mm'}),
                t90          = xr.DataArray(t90.data, dims=outdims, attrs={'units':'°C'}),
                t99          = xr.DataArray(t99.data, dims=outdims, attrs={'units':'°C'}),
                tpcorr       = xr.DataArray(tpcorr.data, dims=outdims, attrs={'units':'unitless'}),
                freezethaw   = xr.DataArray(freezethaw.data, dims=outdims, attrs={'units':'days'}),

                djf_t        = xr.DataArray((seasavg['DJF']['tas'].mean(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                mam_t        = xr.DataArray((seasavg['MAM']['tas'].mean(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                jja_t        = xr.DataArray((seasavg['JJA']['tas'].mean(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                son_t        = xr.DataArray((seasavg['SON']['tas'].mean(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                ann_t        = xr.DataArray((seasavg['ANN']['tas'].mean(dim='time')).data, dims=outdims, attrs={'units':'°C'}),

                djf_t_iav    = xr.DataArray((seasavg['DJF']['tas'].std(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                mam_t_iav    = xr.DataArray((seasavg['MAM']['tas'].std(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                jja_t_iav    = xr.DataArray((seasavg['JJA']['tas'].std(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                son_t_iav    = xr.DataArray((seasavg['SON']['tas'].std(dim='time')).data, dims=outdims, attrs={'units':'°C'}),
                ann_t_iav    = xr.DataArray((seasavg['ANN']['tas'].std(dim='time')).data, dims=outdims, attrs={'units':'°C'}),

                djf_p        = xr.DataArray((seasavg['DJF']['pr'].mean(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                mam_p        = xr.DataArray((seasavg['MAM']['pr'].mean(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                jja_p        = xr.DataArray((seasavg['JJA']['pr'].mean(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                son_p        = xr.DataArray((seasavg['SON']['pr'].mean(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                ann_p        = xr.DataArray((seasavg['ANN']['pr'].mean(dim='time')).data, dims=outdims, attrs={'units':'mm'}),

                djf_p_iav    = xr.DataArray((seasavg['DJF']['pr'].std(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                mam_p_iav    = xr.DataArray((seasavg['MAM']['pr'].std(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                jja_p_iav    = xr.DataArray((seasavg['JJA']['pr'].std(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                son_p_iav    = xr.DataArray((seasavg['SON']['pr'].std(dim='time')).data, dims=outdims, attrs={'units':'mm'}),
                ann_p_iav    = xr.DataArray((seasavg['ANN']['pr'].std(dim='time')).data, dims=outdims, attrs={'units':'mm'}),

                ann_snow     = xr.DataArray(snowaccum.mean(dim='time').data, dims=outdims, attrs={'units':'mm'}),
                ann_snow_iav = xr.DataArray(snowaccum.std(dim='time').data, dims=outdims, attrs={'units':'mm'}),

                wet_day_frac = xr.DataArray(wet_day_frac.data, dims=outdims, attrs={'units':'fraction'}),

                drought_1yr  = xr.DataArray(drought_1yr.data, dims=outdims, attrs={'units':'count'}),
                drought_2yr  = xr.DataArray(drought_2yr.data, dims=outdims, attrs={'units':'count'}),
                drought_5yr  = xr.DataArray(drought_5yr.data, dims=outdims, attrs={'units':'count'}),
            ),
            coords = outcoords,
            attrs = dict(
                description=('Metrics for dataset: ' + outstr))
        )
     
    varlist = list(metrics.variables)
    for remvar in ['ens', 'lat', 'lon']:
        if remvar in varlist:
            varlist.remove(remvar)

    for var in varlist:
        tmp = metrics[var].data
        tmp[np.isnan(metrics['n34pr'].data)] = np.nan
        metrics[var] = (['lat', 'lon'], tmp)
    metrics.to_netcdf(diro+filo, mode='w')


def Compute_Regional_DS_Metrics(region, wtpr_metric_maps, imod=None, imeth=None, GARDLENS=False, OVERWRITE=False):
    
    regions, bukods = make_buko_region_map('icargrid', COMPUTE=False)
    regind = regions.index(region)
    regname = region.replace(' ', '')
    # westregs = ['Pacific Northwest', 'Pacific Southwest', 'Mountain West', 'Desert Southwest']

    gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
    gridtar['lon'] = gridtar['lon'] - 360
    obsmet_diri = f'/glade/derecho/scratch/nlybarger/WT_centroids_era5/wtpr_metric_maps/'
    obsmet_fili = f'{regname}_wtpr_obs_metric_maps_1981-2016.nc'
    obs_wtpr_metric_maps = xr.open_dataset(obsmet_diri + obsmet_fili)
    obsdsetlist = ['PRISM', 'CONUS404', 'Livneh', 'NLDAS', 'GMET', 'nClimGrid', 'gridMET']
    nobs = len(obsdsetlist)
    
    if not GARDLENS:
        mod, meth,_,_,_ = frdc5.get_ds_metavars(imod=imod, imeth=imeth)
        print('Beginning computation of metrics for:')
        print(f'===| Method: {meth}')
        # Checks for valid model/method combination
        if not frdc5.check_modmeth(mod, meth, reg=region):
            return
        
        maskdir = f'/glade/work/nlybarger/downscaling_metrics/cmip5/'
        maskfil = f'CanESM2.MACA.hist.1981-2016.ds.conus.metric.maps.nc'
        maskds = xr.open_dataset(maskdir + maskfil)['pr90']

        diro = f'/glade/work/nlybarger/downscaling_metrics/cmip5/buko_regions/{regname}/'
        filo = f'{mod}.{meth}.hist.1981-2016.ds.{regname}.metrics.nc'

        diri = '/glade/work/nlybarger/downscaling_metrics/cmip5/'
        fili = f'{mod}.{meth}.hist.1981-2016.ds.conus.metric.maps.nc'

        outdims = ['obs']
        outcoords = dict(obs = (['obs'], obsdsetlist))
        destr = f'Metrics for downscaled CMIP5 dataset: {meth} - {mod}'
        ds = xr.open_dataset(diri + fili)
        ds = xr.where(~np.isnan(maskds), ds, np.nan)
        wtpr_metric_maps['mask'] = xr.where(np.isnan(wtpr_metric_maps['wtpr_rmse_ds'][0,1,5,:,:]), np.nan, 1)
    else:
        mod = 'CESM2-LE'
        meth = 'GARDLENS'
        diro = f'/glade/work/nlybarger/downscaling_metrics/GARDLENS/buko_regions/'
        filo = f'GARDLENS.hist.1981-2016.ds.{regname}.metrics.nc'

        diri = '/glade/work/nlybarger/downscaling_metrics/GARDLENS/'
        filis = sorted(glob.glob(f'{diri}GARDLENS.*.hist.1981-2016.ds.conus.metric.maps.nc'))

        enslist = frdc5.CESM2_LE_ensemble_list(GARDLENS=False)
        nens = len(enslist)
        outdims = ['ens', 'obs']
        outcoords = dict(obs = (['obs'], obsdsetlist), ens = (['ens'], enslist))
        destr = 'Metrics for GARDLENS - CESM2-LE'

        for iens,ens in enumerate(enslist):
            if iens == 0:
                ds = xr.open_dataset(filis[iens])
                ds = ds.expand_dims({'ens':[ens]})
            else:
                tmp = xr.open_dataset(filis[iens])
                tmp = tmp.expand_dims({'ens':[ens]})
                ds = xr.concat([ds, tmp], dim='ens')
                del tmp
        wtpr_metric_maps['mask'] = xr.where(np.isnan(wtpr_metric_maps['wtpr_rmse_ds'][0,0,:,:]), np.nan, 1)

    ## Read in downscaled data and mask out everything except the region of interest

    nci = xr.where(bukods['regions'] == regind+1, ds, np.nan)
    del ds
    nci = xr.where(np.isnan(bukods['mask']), np.nan, nci)

    
    regridder = xesmf.Regridder(bukods, wtpr_metric_maps, 'nearest_s2d')
    bukods_regridded = regridder(bukods)
    bukods_regridded = xr.where(np.isnan(wtpr_metric_maps['mask']), np.nan, bukods_regridded)
    wtpr_metric_maps = xr.where(bukods_regridded['regions'] == regind+1, wtpr_metric_maps, np.nan)
    # Set all methods to nan where MACA is nan
    if not GARDLENS:
        # Expand the mask to all methods using broadcasting
        expanded_mask = wtpr_metric_maps['mask'].broadcast_like(wtpr_metric_maps)

        # Set all methods to NaN wherever MACA/CanESM2 is NaN
        wtpr_metric_maps = xr.where(~np.isnan(expanded_mask), wtpr_metric_maps, np.nan)
    # wtpr_metric_maps.to_netcdf(f'/glade/work/nlybarger/downscaling_metrics/{regname}_test_wtpr_metric_maps.nc', mode='w')

    # if region in westregs:
    #     # For western regions, limit to ICARwest domain (mostly affects Desert Southwest)
    #     tmpdir = '/glade/work/nlybarger/downscaling_metrics/cmip5/'
    #     tmpfil = 'CanESM2.ICARwest.hist.1981-2016.ds.conus.metric.maps.nc'
    #     tmp = xr.open_dataset(tmpdir + tmpfil)
    #     if tmp['lon'].min() > 0:
    #         tmp['lon'] = tmp['lon'] - 360
    #     nci = xr.where(~np.isnan(tmp['pr90']), nci, np.nan)
    #     del tmpdir
    #     del tmpfil
    #     del tmp
    #     tmpdir = '/glade/derecho/scratch/nlybarger/WT_centroids_era5/wtpr_metric_maps/'
    #     tmpfil = f'{tmpdir}{regname}_wtpr_DS_metric_maps_1981-2016.nc'
    #     tmp = xr.open_dataset(tmpfil)
    #     tmp = tmp.rename({'method':'methods'}).sel(methods='ICARwest', model='CanESM2')
    #     wtpr_metric_maps = xr.where(~np.isnan(tmp['wtpr_rmse_ds']), wtpr_metric_maps, np.nan)
    #     del tmpdir
    #     del tmpfil
    #     del tmp

    # del ds
    # del bukods
    # del bukods_regridded
    # del regridder

    if os.path.exists(diro+filo) and OVERWRITE:
        os.remove(diro+filo)
    elif os.path.exists(diro+filo) and not OVERWRITE:
        return
    
    print('===|===| Model: ' + mod)
    print('===|===|===| Region: ' + region)

    tmp_varlist = list(nci.variables)
    varlist = [item for item in tmp_varlist if item not in ['ens', 'lat', 'lon', 'time']]

    obsmaps = {}

    for obs in obsdsetlist:
        f = '/glade/work/nlybarger/downscaling_metrics/obs/' + obs + '.1981-2016.ds.conus.metric.maps.nc'
        obsmaps[obs] = xr.open_dataset(f)
        obsmaps[obs]['lon'] = nci['lon']
        obsmaps[obs]['lat'] = nci['lat']

    mets = {}
    if not GARDLENS:
        for var in varlist:
            mets[var + '_r'] = np.full(nobs, np.nan)
            mets[var + '_rmse'] = np.full(nobs, np.nan)
            for i in range(nobs):
                mets[var + '_r'][i] = xs.pearson_r(nci[var], obsmaps[obsdsetlist[i]][var], skipna=True)
                mets[var + '_rmse'][i] = xs.rmse(nci[var], obsmaps[obsdsetlist[i]][var], skipna=True)
        # WT metrics
        daytoday_rmse = np.full((nobs, 6), np.nan)
        daytoday_scorr = np.full((nobs, 6), np.nan)
        clim_rmse = np.full((nobs, 6), np.nan)
        clim_scorr = np.full((nobs, 6), np.nan)
        for iobs, obs in enumerate(obsdsetlist):
            for iwt in range(6):
                tmpobs_rmse = obs_wtpr_metric_maps['wtpr_rmse_obs'].sel(WT=iwt, obs=obs)
                tmpobs_clim = obs_wtpr_metric_maps['wtpr_clim_obs'].sel(WT=iwt, obs=obs)
                tmpds_rmse = wtpr_metric_maps['wtpr_rmse_ds'].sel(WT=iwt, model=mod, methods=meth)
                tmpds_clim  = wtpr_metric_maps['wtpr_clim_ds'].sel(WT=iwt, model=mod, methods=meth)

                daytoday_rmse[iobs, iwt]  = xs.rmse(tmpobs_rmse, tmpds_rmse, skipna=True)
                daytoday_scorr[iobs, iwt] = xs.pearson_r(tmpobs_rmse, tmpds_rmse, skipna=True)
                clim_rmse[iobs, iwt]      = xs.rmse(tmpobs_clim, tmpds_clim, skipna=True)
                clim_scorr[iobs, iwt]     = xs.pearson_r(tmpobs_clim, tmpds_clim, skipna=True)
    else:
        for var in varlist:
            mets[var + '_r'] = np.full((nens, nobs), np.nan)
            mets[var + '_rmse'] = np.full((nens, nobs), np.nan)
            for iobs in range(nobs):
                for iens, ens in enumerate(enslist):
                    mets[var + '_r'][iens, iobs]    = xs.pearson_r(nci[var].sel(ens=ens), obsmaps[obsdsetlist[iobs]][var], skipna=True)
                    mets[var + '_rmse'][iens, iobs] = xs.rmse(nci[var].sel(ens=ens), obsmaps[obsdsetlist[iobs]][var], skipna=True)
        # WT metrics
        daytoday_rmse = np.full((nens, nobs, 6), np.nan)
        daytoday_scorr = np.full((nens, nobs, 6), np.nan)
        clim_rmse = np.full((nens, nobs, 6), np.nan)
        clim_scorr = np.full((nens, nobs, 6), np.nan)
        for iens, ens in enumerate(enslist):
            for iobs, obs in enumerate(obsdsetlist):
                for iwt in range(6):
                    tmpobs_rmse = obs_wtpr_metric_maps['wtpr_rmse_obs'].sel(WT=iwt, obs=obs)
                    tmpobs_clim = obs_wtpr_metric_maps['wtpr_clim_obs'].sel(WT=iwt, obs=obs)
                    tmpds_rmse  = wtpr_metric_maps['wtpr_rmse_ds'].sel(WT=iwt, ens=ens)
                    tmpds_clim  = wtpr_metric_maps['wtpr_clim_ds'].sel(WT=iwt, ens=ens)

                    daytoday_rmse[iens, iobs, iwt]  = xs.rmse(tmpobs_rmse, tmpds_rmse, skipna=True)
                    daytoday_scorr[iens, iobs, iwt] = xs.pearson_r(tmpobs_rmse, tmpds_rmse, skipna=True)
                    clim_rmse[iens, iobs, iwt]      = xs.rmse(tmpobs_clim, tmpds_clim, skipna=True)
                    clim_scorr[iens, iobs, iwt]     = xs.pearson_r(tmpobs_clim, tmpds_clim, skipna=True)


    # Create metrics dataset and save
    data_vars_dict = {}
    for key in list(mets.keys()):
        data_vars_dict[key] = (outdims, mets[key])

    for iwt in range(6):
        wt = str(iwt)
        if not GARDLENS:
            data_vars_dict[f'wt{wt}_daytoday_rmse']  = (outdims, daytoday_rmse[:, iwt])
            data_vars_dict[f'wt{wt}_daytoday_scorr'] = (outdims, daytoday_scorr[:, iwt])
            data_vars_dict[f'wt{wt}_clim_rmse']      = (outdims, clim_rmse[:, iwt])
            data_vars_dict[f'wt{wt}_clim_scorr']     = (outdims, clim_scorr[:, iwt])
        else:
            data_vars_dict[f'wt{wt}_daytoday_rmse']  = (outdims, daytoday_rmse[:, :, iwt])
            data_vars_dict[f'wt{wt}_daytoday_scorr'] = (outdims, daytoday_scorr[:, :, iwt])
            data_vars_dict[f'wt{wt}_clim_rmse']      = (outdims, clim_rmse[:, :, iwt])
            data_vars_dict[f'wt{wt}_clim_scorr']     = (outdims, clim_scorr[:, :, iwt])

    metrics = xr.Dataset(
        data_vars = data_vars_dict,
        coords = outcoords,
        attrs = dict(
            description=(destr)
        )
    )
    metrics.to_netcdf(diro+filo, mode='w')
    return


def gardlens_normalizers(LOAD=True, COMPUTE=False, SAVE=False):
    if LOAD and not COMPUTE:
        diri = '/glade/work/nlybarger/downscaling_metrics/'
        normalizers = xr.open_dataset(f'{diri}/GARDLENS.normalizers.1981-2016.ds.buko_regions.nc')
        return normalizers
    elif COMPUTE:
        regions, _ = make_buko_region_map('icargrid', COMPUTE=False)
        gard_regmets = {}
        diri = '/glade/work/nlybarger/downscaling_metrics/GARDLENS/buko_regions/'
        for reg in regions:
            regname = reg.replace(' ', '')
            gard_regmets[reg] = xr.open_dataset(f'{diri}/GARDLENS.hist.1981-2016.ds.{regname}.metrics.nc')
        gard_metlist = list(gard_regmets[reg].data_vars)
        for var in ['obs', 'ens']:
            if var in gard_metlist:
                gard_metlist.remove(var)

        obsstd = {}
        ensstd = {}
        combstd = {}
        obsstd = np.zeros((len(regions), len(gard_metlist)))
        ensstd = np.zeros((len(regions), len(gard_metlist)))
        combstd = np.zeros((len(regions), len(gard_metlist)))
        for ireg, reg in enumerate(regions):
            for imet, met in enumerate(gard_metlist):
                tmpmet = gard_regmets[reg][met].copy()
                obsstd[ireg,imet] = tmpmet.std(dim='obs').mean(dim='ens')
                ensstd[ireg,imet] = tmpmet.std(dim='ens').mean(dim='obs')
                combstd[ireg,imet] = np.sqrt(np.square(obsstd[ireg,imet]) + np.square(ensstd[ireg,imet]))

        normalizers = xr.Dataset(
            data_vars={
                'obsstd': (('region', 'metric'), obsstd),
                'ensstd': (('region', 'metric'), ensstd),
                'combstd': (('region', 'metric'), combstd),
            },
            coords={
                'region': regions,
                'metric': gard_metlist
            },
        )
        diro = '/glade/work/nlybarger/downscaling_metrics'

        # Identify outliers and regress them toward the mean for each metric (by end string after '_')
        # for met in gard_metlist:
        #     met_suffix = met.split('_')[-1]
        #     if met_suffix == 'scorr':
        #         met_suffix = 'r'
            # Find all metrics with the same suffix
        # for suffix in ['r', 'rmse']:
        #     if suffix == 'r':
        #         same_suffix_indices = [i for i, m in enumerate(gard_metlist) if (m.endswith(suffix) or m.endswith('scorr'))]
        #     else:
        #         same_suffix_indices = [i for i, m in enumerate(gard_metlist) if m.endswith(suffix)]
        #     # For each region, check for outliers in these metrics (only within the same region)
        #     for ireg in range(len(regions)):
        #         for var in ['obsstd', 'ensstd']:
        #             vals = np.array([normalizers[var].data[ireg, idx] for idx in same_suffix_indices])
        #             mean_val = np.nanmean(vals)
        #             # Regress outliers (order of magnitude difference from mean) toward mean
        #             for idx in same_suffix_indices:
        #                 val = normalizers[var].data[ireg, idx]
        #                 if mean_val != 0 and (val > 0 and np.abs(np.log10(val) - np.log10(mean_val)) > 1.5):
        #                     print(f'Regressing outlier in {var} for region {regions[ireg]}, metric {gard_metlist[idx]}: {val} toward mean {mean_val}')
        #                     # Move value halfway toward mean
        #                     normalizers[var].data[ireg, idx] = (val + mean_val) / 2
        # combstd = np.sqrt(np.square(normalizers['obsstd']) + np.square(normalizers['ensstd']))
        # normalizers['combstd'] = (('region', 'metric'), combstd.data)
        if SAVE:
            normalizers.to_netcdf(f'{diro}/GARDLENS.normalizers.1981-2016.ds.buko_regions.nc')
        if LOAD:
            return normalizers
        return
