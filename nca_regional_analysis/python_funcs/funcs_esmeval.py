import xarray as xr
import numpy as np
import glob
import funcs_general as fg
import pandas as pd
import xskillscore as xs
import os
import pickle
import scipy.stats
import xesmf

# NCA region list:
# Northwest, Southwest, Northern Great Plains, Southern Great Plains, Midwest, Southeast, Northeast

def spatial_average(dataArray,latstr,lonstr,latmin=None,latmax=None,lonmin=None,lonmax=None):
    # nca_region_dict = {'Northeast': 1, 'Southeast': 2, 'Midwest': 4,
    #             'Northern Great Plains': 5, 'Southern Great Plains': 6, 
    #             'Southwest': 7, 'Northwest': 8}

    # if domain in nca_region_dict:
    #     nca_mask = xr.open_dataset('/glade/work/nlybarger/data/NCA_region_masks/NCA_region_mask_1deg_esmeval.nc')
    #     mask = xr.where(nca_mask['region_mask'] == nca_region_dict[domain],1,np.nan)
    #     datout = dataArray.where(mask == 1,drop=True)
    # elif domain == 'global':
    #     datout = dataArray.copy()
    # elif latmin != None and latmax != None and lonmin != None and lonmax != None:
    #     datout = dataArray.where((dataArray[latstr] >= latmin) & (dataArray[latstr] <= latmax) &
    #                             (dataArray[lonstr] >= lonmin) & (dataArray[lonstr] <= lonmax),drop=True)
    # else:
    #     raise ValueError('Domain not recognized')

    weights = np.cos(np.deg2rad(dataArray[latstr]))
    weights.name = 'weights'
    datout_weighted = dataArray.weighted(weights).mean(dim=[latstr,lonstr],skipna=True)
    return datout_weighted


def read_cmip_data(fili, domain, latmin=None, latmax=None, lonmin=None, lonmax=None):
    nca_region_dict = {'Northeast': 1, 'Southeast': 2, 'Midwest': 4,
            'Northern Great Plains': 5, 'Southern Great Plains': 6, 
            'Southwest': 7, 'Northwest': 8}
    nca_regions = list(nca_region_dict.keys())
    odiri = '/glade/campaign/ral/hap/nlybarger/OBS/'
    if domain == 'global':
        obsgrid = xr.open_dataset(f'{odiri}CRU/1deg.cru.global.p.t.nc',engine='netcdf4')
    elif (domain == 'conus') or (domain in nca_regions):
        obsgrid = xr.open_dataset(f'{odiri}PRISM/1deg.prism.conus.p.t.nc',engine='netcdf4')
        if domain in nca_regions:
            nca_mask = xr.open_dataset('/glade/work/nlybarger/data/NCA_region_masks/NCA_region_mask_1deg_esmeval.nc')
            mask = xr.where(nca_mask['region_mask'] == nca_region_dict[domain],1,np.nan)
            obsgrid = obsgrid.copy().where(mask == 1,drop=True)
    elif (domain == 'custom'):
        if latmin != None and latmax != None and lonmin != None and lonmax != None:
            obsgrid = xr.open_dataset(f'{odiri}CRU/1deg.cru.global.p.t.nc',engine='netcdf4')
            obsgrid = obsgrid.where((obsgrid['lat'] >= latmin) & (obsgrid['lat'] <= latmax) &
                                            (obsgrid['lon'] >= lonmin) & (obsgrid['lon'] <= lonmax),drop=True)
        else:
            raise ValueError('Custom domain specified but no lat/lon bounds given')
    else:
        raise ValueError('Domain not recognized')
    
    gridout = xr.Dataset(
        data_vars = dict(
            mask=(['lat','lon'],xr.where(~np.isnan(obsgrid['pr'].isel(time=0)), 1, 0).data),
        ),
        coords = dict(
            lat=(['lat'],obsgrid['lat'].data),
            lon=(['lon'],obsgrid['lon'].data),
            time=(['time'],obsgrid['time'].data),
        ),
    )

    tmp = xr.open_dataset(fili)
    try:
        regridder = xesmf.Regridder(tmp, gridout, 'bilinear')
    except ValueError:
        regridder = xesmf.Regridder(tmp, gridout, 'bilinear', ignore_degenerate=True)
        
    
    variants = list(tmp['ens'].data)
    nci = regridder(tmp)

    nci['n34'] = (['ens','time'],tmp['n34'].data)
    nci['eli'] = (['ens','time'],tmp['eli'].data)
    return nci, variants


def read_obs_climate_data(obsdsetlist, fnames, domain, latmin=None, latmax=None, lonmin=None, lonmax=None):
    nca_region_dict = {'Northeast': 1, 'Southeast': 2, 'Midwest': 4,
            'Northern Great Plains': 5, 'Southern Great Plains': 6, 
            'Southwest': 7, 'Northwest': 8}
    nca_regions = list(nca_region_dict.keys())
    odiri = '/glade/campaign/ral/hap/nlybarger/OBS/'
    datout = {}
    oyears = {}
    for iobs, obs in enumerate(obsdsetlist):
        if domain == 'global':
            datout[obs] = xr.open_dataset(f'{odiri}{obs}/1deg.{fnames[iobs]}.global.p.t.nc',engine='netcdf4')
        elif (domain == 'conus') or (domain in nca_regions):
            datout[obs] = xr.open_dataset(f'{odiri}{obs}/1deg.{fnames[iobs]}.conus.p.t.nc',engine='netcdf4')
            if domain in nca_regions:
                nca_mask = xr.open_dataset('/glade/work/nlybarger/data/NCA_region_masks/NCA_region_mask_1deg_esmeval.nc')
                mask = xr.where(nca_mask['region_mask'] == nca_region_dict[domain],1,np.nan)
                datout[obs] = datout[obs].copy().where(mask == 1,drop=True)
        elif (domain == 'custom'):
            if latmin != None and latmax != None and lonmin != None and lonmax != None:
                datout[obs] = xr.open_dataset(f'{odiri}{obs}/1deg.{fnames[iobs]}.global.p.t.nc',engine='netcdf4')
                datout[obs] = datout[obs].where((datout[obs]['lat'] >= latmin) & (datout[obs]['lat'] <= latmax) &
                                                (datout[obs]['lon'] >= lonmin) & (datout[obs]['lon'] <= lonmax),drop=True)
            else:
                raise ValueError('Custom domain specified but no lat/lon bounds given')
        else:
            raise ValueError('Domain not recognized')
        oyears[obs] = list(datout[obs].groupby('time.year').groups)
    n34f = '/glade/work/nlybarger/data/clim_indices/nino34.1870-2021.txt'
    fp = open(n34f,'r')
    n34o = np.genfromtxt(fp,delimiter=',',usecols=np.arange(1,13),dtype='f4')
    n34o = np.reshape(n34o[30:150,:],(120*12))
    fp.close()

    elifi = '/glade/work/nlybarger/data/clim_indices/ELI_ERSSTv5_1854.01-2019.12.csv'
    fp = open(elifi,'r')
    elio = np.genfromtxt(fp,delimiter=',',usecols=np.arange(47,167),dtype='f4',skip_header=1)
    elio = np.transpose(elio)
    elio = np.reshape(elio,(120*12,))
    fp.close()

    indy = xr.Dataset(
            data_vars = dict(
                eli=(['time'], elio),
                n34=(['time'], n34o),
            ),
            coords = dict(
                time=(['time'], pd.date_range('1900-01-01','2019-12-31',freq='MS')),
            ),
    )

    oseasvars = {}
    for obs in obsdsetlist:
        # print(f'Beginning to read observational data from: {obs}')
        if obs in ['CRU','ERA-5','GMET','PRISM']:
            oyears[obs] = oyears[obs][:-2]

        datout[obs] = datout[obs].sel(time=slice(str(oyears[obs][0])+'-01-01',str(oyears[obs][-1])+'-12-31'))
        datout[obs]['n34'] = (['time'],indy['n34'].sel(time=slice(f'{oyears[obs][0]}-01-01',f'{oyears[obs][-1]}-12-31')).data)
        datout[obs]['eli'] = (['time'],indy['eli'].sel(time=slice(f'{oyears[obs][0]}-01-01',f'{oyears[obs][-1]}-12-31')).data)
        oseasvars[obs] = seasonal_avg_vars_obs(datout[obs],obs,'lat','lon',True)
    return datout, oyears, oseasvars


def compute_obs_metrics(datin, obsdsetlist, oyears, oseasvars, domain, OVERWRITE=False):
    diro = '/glade/work/nlybarger/data/esmeval/obs_metrics/'
    ncdiro = '/glade/work/nlybarger/ESM_eval_metrics/obs_metrics/'

    dom = domain.replace(" ", "")
    filo_all = f'{dom}.obs.metrics.all.pkl'
    filo_mean = f'{dom}.obs.metrics.mean.pkl'
    if (os.path.exists(f'{diro}{filo_all}')) and (os.path.exists(f'{diro}{filo_mean}')) and not OVERWRITE:
        print(f'Metrics file for observational datasets already exists. Loading...')
        with open(f'{diro}{filo_all}', 'rb') as f:
            metrics_allobs = pickle.load(f)
        f.close()
        with open(f'{diro}{filo_mean}', 'rb') as f:
            metout = pickle.load(f)
        f.close()
        return metrics_allobs, metout


    print(f'Computing metrics for observational datasets over domain: {domain}')
    metrics_allobs = {}
    metrics_allobs['Mean-T'] = {}
    metrics_allobs['Mean-P'] = {}
    metrics_allobs['Trend-T'] = {}
    metrics_allobs['Trend-P'] = {}

    if domain != 'global':
        metrics_allobs['SeasAmp-T'] = {}
        metrics_allobs['SeasAmp-P'] = {}
    nobs = len(obsdsetlist)

    seasdict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    seasons = list(seasdict.keys())
    for seas in seasons:
        metrics_allobs[seas+'-T'] = {}
        metrics_allobs[seas+'-P'] = {}

    ensomets = ['Nino3.4-T', 'ELI-T', 'Nino3.4-P', 'ELI-P']
    for met in ensomets:
        metrics_allobs[met] = {}


    for obs in obsdsetlist:
        nyr = len(oyears[obs])

        # Mean annual temperature and precipitation
        gbto = spatial_average(datin[obs]['tas'],'lat','lon')
        metrics_allobs['Mean-T'][obs] = spatial_average(datin[obs]['tas'].groupby('time.year').mean(dim='time',skipna=True),'lat','lon')

        tmp = datin[obs]['pr']/10
        gbpo = spatial_average(tmp,'lat','lon')
        metrics_allobs['Mean-P'][obs] = spatial_average(tmp.groupby('time.year').sum(dim='time',skipna=False),'lat','lon')

        metrics_allobs['Mean-T'][obs] = metrics_allobs['Mean-T'][obs].mean(dim='year')
        metrics_allobs['Mean-P'][obs] = metrics_allobs['Mean-P'][obs].mean(dim='year')

        if domain != 'global':
            # Seasonal amplitude of temperature
            metrics_allobs['SeasAmp-T'][obs] = np.zeros(nyr)
            for it, year in enumerate(oyears[obs]):
                tmp = gbto.copy().sel(time=slice(f'{year}-01-01',f'{year}-12-31'))
                metrics_allobs['SeasAmp-T'][obs][it] = tmp.max()-tmp.min()
            metrics_allobs['SeasAmp-T'][obs] = metrics_allobs['SeasAmp-T'][obs].mean()
            del tmp

            # Seasonal amplitude of precipitation
            metrics_allobs['SeasAmp-P'][obs] = np.zeros(nyr)
            for it, year in enumerate(oyears[obs]):
                tmp = gbpo.copy().sel(time=slice(f'{year}-01-01',f'{year}-12-31'))
                metrics_allobs['SeasAmp-P'][obs][it] = tmp.max()-tmp.min()
            metrics_allobs['SeasAmp-P'][obs] = metrics_allobs['SeasAmp-P'][obs].mean()
            del tmp


        
        # Precipitation/Temperature trends (UDel and CRU only)
        if obs in ['CRU','UDel']:
            tmpt = datin[obs]['tas'].copy().sel(time=slice('1901-01-01','2014-12-30'),drop=True)
            tmpp = datin[obs]['pr'].copy().sel(time=slice('1901-01-01','2014-12-30'),drop=True)/10
            ttmpo = spatial_average(tmpt.copy().groupby('time.year').mean(dim='time',skipna=True),'lat','lon')
            ptmpo = spatial_average(tmpp.copy().groupby('time.year').sum(dim='time',skipna=False),'lat','lon')

            metrics_allobs['Trend-T'][obs] = (xs.linslope(ttmpo['year'], ttmpo, dim='year')*100).data
            metrics_allobs['Trend-P'][obs] = (xs.linslope(ptmpo['year'], ptmpo, dim='year')*100).data


        # Seasonal mean temperature and precipitation maps
        for seas in seasons:
            metrics_allobs[seas+'-T'][obs] = datin[obs]['tas'].groupby('time.month').mean(dim='time').sel(month=seasdict[seas],drop=True).mean(dim='month')
        for seas in seasons:
            metrics_allobs[seas+'-P'][obs] = datin[obs]['pr'].groupby('time.month').mean(dim='time').sel(month=seasdict[seas],drop=True).mean(dim='month')

        # Nino3.4/ELI - variable Anomalies DJF
        metrics_allobs['Nino3.4-T'][obs] = xs.pearson_r(oseasvars[obs]['DJF']['n34'],oseasvars[obs]['DJF']['tasanom'],dim='time')
        metrics_allobs['ELI-T'][obs] = xs.pearson_r(oseasvars[obs]['DJF']['eli'],oseasvars[obs]['DJF']['tasanom'],dim='time')
        metrics_allobs['Nino3.4-P'][obs] = xs.pearson_r(oseasvars[obs]['DJF']['n34'],oseasvars[obs]['DJF']['pranom'],dim='time')
        metrics_allobs['ELI-P'][obs] = xs.pearson_r(oseasvars[obs]['DJF']['eli'],oseasvars[obs]['DJF']['pranom'],dim='time')

    

    # Now compute means across observational datasets
    # For maps, compute each obs against each other obs then average
    # For simple numbers, just average the value
    if domain == 'global':
        # Don't compute seasonal amplitude for global
        simpmets = ['Mean-T', 'Mean-P', 'Trend-T', 'Trend-P']
    else:
        simpmets = ['Mean-T', 'Mean-P', 'SeasAmp-T', 'SeasAmp-P', 'Trend-T', 'Trend-P']
    
    metout = {}
    for met in simpmets:
        metout[met] = np.zeros(1)
        ndt = 0
        if met in ['Trend-P','Trend-T']:
            for obs in ['CRU', 'UDel']:
                metout[met] += metrics_allobs[met][obs].data
                ndt += 1
        else:
            for obs in obsdsetlist:
                metout[met] += metrics_allobs[met][obs].data
                ndt += 1
        metout[met] = metout[met]/ndt

    # ensomets defined above in the obs loop
    for met in ensomets:
        metrics_allobs[met+'-r'] = np.full((nobs,nobs), np.nan)
        for i, obs in enumerate(obsdsetlist):
            for j, obs2 in enumerate(obsdsetlist):
                if j==i:
                    continue
                else:
                    metrics_allobs[met+'-r'][i,j] = xs.pearson_r(metrics_allobs[met][obs],metrics_allobs[met][obs2],skipna=True).data
        metrics_allobs[met+'-r'] = np.nanmean(metrics_allobs[met+'-r'], axis=1)

    if domain == 'global':
        metout['DJF ELI Median'] = np.median(oseasvars['CRU']['DJF']['eli'])

    # For seasonal average correlations and standard deviations, compute each obs against each other obs then average

    for seas in seasons:
        metrics_allobs['SpaceCorr '+seas+'-T'] = np.full((nobs,nobs),np.nan)
        metrics_allobs['SpaceCorr '+seas+'-P'] = np.full((nobs,nobs),np.nan)
        metrics_allobs['SpaceSD '+seas+'-T'] = np.full((nobs,nobs),np.nan)
        metrics_allobs['SpaceSD '+seas+'-P'] = np.full((nobs,nobs),np.nan)
        for i, obs in enumerate(obsdsetlist):
            for j, obs2 in enumerate(obsdsetlist):
                if j==i:
                    continue
                else:
                    metrics_allobs['SpaceCorr '+seas+'-T'][i,j] = xs.pearson_r(metrics_allobs[seas+'-T'][obs],metrics_allobs[seas+'-T'][obs2],skipna=True).data
                    metrics_allobs['SpaceCorr '+seas+'-P'][i,j] = xs.pearson_r(metrics_allobs[seas+'-P'][obs],metrics_allobs[seas+'-P'][obs2],skipna=True).data
                    metrics_allobs['SpaceSD '+seas+'-T'][i,j] = np.nanstd(metrics_allobs[seas+'-T'][obs].data)/np.nanstd(metrics_allobs[seas+'-T'][obs2].data)
                    metrics_allobs['SpaceSD '+seas+'-P'][i,j] = np.nanstd(metrics_allobs[seas+'-P'][obs].data)/np.nanstd(metrics_allobs[seas+'-P'][obs2].data)
        metrics_allobs['SpaceCorr '+seas+'-T'] = np.nanmean(metrics_allobs['SpaceCorr '+seas+'-T'], axis=1)
        metrics_allobs['SpaceCorr '+seas+'-P'] = np.nanmean(metrics_allobs['SpaceCorr '+seas+'-P'], axis=1)
        metrics_allobs['SpaceSD '+seas+'-T'] = np.nanmean(metrics_allobs['SpaceSD '+seas+'-T'], axis=1)
        metrics_allobs['SpaceSD '+seas+'-P'] = np.nanmean(metrics_allobs['SpaceSD '+seas+'-P'], axis=1)

    spacemets = ['Nino3.4-T', 'Nino3.4-P', 
                 'ELI-T', 'ELI-P', 
                 'DJF-T', 'DJF-P', 
                 'MAM-T', 'MAM-P', 
                 'JJA-T', 'JJA-P', 
                 'SON-T', 'SON-P']
    
    # Create datasets from metout

    spacemet_ds = xr.Dataset(
        {
            met: xr.DataArray(
                np.stack([metrics_allobs[met][obs].values for obs in (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else metrics_allobs['DJF-T'].keys())]),
                dims=['obs', 'lat', 'lon'],
                coords={
                    'obs': (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else list(metrics_allobs['DJF-T'].keys())),
                    'lat': metrics_allobs[met][next(iter(metrics_allobs[met]))].coords['lat'],
                    'lon': metrics_allobs[met][next(iter(metrics_allobs[met]))].coords['lon']
                }
            )
            for met in spacemets
        }
    )
    spacemet_ds.to_netcdf(f'{ncdiro}obs.{dom}.metric.maps.nc')

    othermets = []
    for met in metrics_allobs.keys():
        if met not in spacemets:
            othermets.append(met)

    othermets_ds = xr.Dataset()

    for met in othermets:
        try:
            if hasattr(metrics_allobs[met]['CRU'], 'data') and hasattr(metrics_allobs[met]['CRU'], 'dims'):
                othermets_ds[met] = xr.DataArray(
                    np.stack([metrics_allobs[met][obs].values for obs in (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else metrics_allobs['DJF-T'].keys())]),
                    dims=['obs'],
                    coords={
                        'obs': (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else list(metrics_allobs['DJF-T'].keys())),
                    }
                )
            else:
                # If metrics[met][obs] is a numpy array, just stack directly
                othermets_ds[met] = xr.DataArray(
                    np.stack([metrics_allobs[met][obs] for obs in (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else metrics_allobs['DJF-T'].keys())]),
                    dims=['obs'],
                    coords={
                    'obs': (['CRU', 'UDel'] if met in ['Trend-T', 'Trend-P'] else list(metrics_allobs['DJF-T'].keys())),
                    }
                )
        except IndexError:
                othermets_ds[met] = xr.DataArray(
                    metrics_allobs[met],
                    dims=['obs'],
                    coords={
                    'obs': (list(metrics_allobs['DJF-T'].keys())),
                    }
                )
    othermets_ds.to_netcdf(f'{ncdiro}obs.{dom}.metrics.nc')


    # Save metrics to pickle file
    with open(f'{diro}{filo_all}', 'wb') as f:
        pickle.dump(metrics_allobs, f)
    f.close()
    with open(f'{diro}{filo_mean}', 'wb') as f:
        pickle.dump(metout, f)
    f.close()

    return metrics_allobs, metout


def compute_model_metrics(datin, model, variants, domain, obs_seasvars, metrics_allobs, cmip, OVERWRITE=False, LOAD=False):
    nca_regions = ['Northeast', 'Southeast', 'Midwest', 'Northern Great Plains',
               'Southern Great Plains', 'Southwest', 'Northwest']
    if domain in nca_regions:
        dom = domain.replace(" ", "")
        diro = f'/glade/work/nlybarger/data/esmeval/{cmip}_metrics/NCA_regions/{dom}/'
        ncdiro = f'/glade/work/nlybarger/ESM_eval_metrics/{cmip}_metrics/NCA_regions/{dom}/'
    else:
        dom = domain.replace(" ", "")
        diro = f'/glade/work/nlybarger/data/esmeval/{cmip}_metrics/{domain}/'
        ncdiro = f'/glade/work/nlybarger/ESM_eval_metrics/{cmip}_metrics/{domain}/'
    filo = f'{model}.{cmip}.metrics.{dom}.pkl'
    if (os.path.exists(f'{diro}{filo}')) and (not OVERWRITE) and LOAD:
        print(f'Metrics file for {model} already exists. Loading...')
        with open(f'{diro}{filo}', 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    elif (os.path.exists(f'{diro}{filo}')) and (not OVERWRITE) and LOAD:
        print(f'Metrics file for {model} already exists. Skipping...')
        return
    if model == 'NorCPM1':
        print(model + ' raises an error from the ESMF regridder.  Skipping.')
        return
    print(f'Computing metrics for model: {model} over domain: {domain}')

    nvar = len(variants)
    obsdsetlist = list(metrics_allobs['Mean-T'].keys())
    nobs = len(obsdsetlist)
    latstr, lonstr, _, _ = fg.coordnames(datin)

    if datin['tas'].max() > 100.:
        datin['tas'] -= 273.15
    month_length = datin['tas'].time.dt.days_in_month
    datin['pr'] = datin['pr']*month_length

    # =========================================
    # Initialize dictionary to store metrics

    metrics = {}

    seasdict = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    seasons = list(seasdict.keys())

    ensomets = ['Nino3.4-T', 'ELI-T', 'Nino3.4-P', 'ELI-P']

    ## =========================================
    # Mean annual temperature and precipitation
    gbt = spatial_average(datin['tas'],latstr,lonstr)
    metrics['Mean-T'] = spatial_average(datin['tas'].groupby('time.year').mean(dim='time',skipna=True),latstr,lonstr)

    tmp = datin['pr']/10
    gbp = spatial_average(tmp,latstr,lonstr)
    metrics['Mean-P'] = spatial_average(tmp.groupby('time.year').sum(dim='time',skipna=False),latstr,lonstr)

    metrics['Mean-T'] = metrics['Mean-T'].mean(dim='year')
    metrics['Mean-P'] = metrics['Mean-P'].mean(dim='year')

    # Seasonal Amplitude of temperature and precipitation (only for regional domains)
    if domain != 'global':
        years = list(datin.groupby('time.year').groups)
        metrics['SeasAmp-T'] = np.zeros((nvar, len(years)))
        metrics['SeasAmp-P'] = np.zeros((nvar, len(years)))
        for iy,year in enumerate(years):
            tmpT = gbt.copy().sel(time=slice(f'{year}-01-01',f'{year}-12-30'))
            tmpP = gbp.copy().sel(time=slice(f'{year}-01-01',f'{year}-12-30'))
            for ivar, var in enumerate(variants):
                tmpT0 = tmpT.copy().sel(ens=var)
                tmpP0 = tmpP.copy().sel(ens=var)
                metrics['SeasAmp-T'][ivar,iy] = tmpT0.max()-tmpT0.min()
                metrics['SeasAmp-P'][ivar,iy] = tmpP0.max()-tmpP0.min()
                del tmpT0
                del tmpP0
            del tmpT
            del tmpP
        metrics['SeasAmp-T'] = metrics['SeasAmp-T'].mean(axis=1)
        metrics['SeasAmp-P'] = metrics['SeasAmp-P'].mean(axis=1)

    # Precipitation/Temperature trends
    tmpt = datin['tas'].copy().sel(time=slice('1901-01-01','2014-12-30'),drop=True)
    tmpp = datin['pr'].copy().sel(time=slice('1901-01-01','2014-12-30'),drop=True)/10
    ttmpo = spatial_average(tmpt.copy().groupby('time.year').mean(dim='time',skipna=True),'lat','lon')
    ptmpo = spatial_average(tmpp.copy().groupby('time.year').sum(dim='time',skipna=False),'lat','lon')

    metrics['Trend-T'] = (xs.linslope(ttmpo['year'], ttmpo, dim='year')*100).data
    metrics['Trend-P'] = (xs.linslope(ptmpo['year'], ptmpo, dim='year')*100).data

    # Compute seasonal mean variables
    seasvars = seasonal_avg_vars_model(datin,model,latstr,lonstr,ELI=True)

    # ELI median bias and Levene's statistic
    if domain == 'global':
        metrics['DJF ELI Median'] = np.zeros(nvar)
        metrics['DJF ELI LevStat'] = np.zeros(nvar)
        for ivar, var in enumerate(variants):
            metrics['DJF ELI Median'][ivar] = np.median(seasvars['DJF']['eli'].sel(ens=var))
            metrics['DJF ELI LevStat'][ivar],_ = scipy.stats.levene(seasvars['DJF']['eli'].sel(ens=var).data,
                                                obs_seasvars['CRU']['DJF']['eli'].data,center='median')
        
    # Correlations between ELI/N34 pr/tas maps
    # Nino3.4/ELI - variable Anomalies DJF
    metrics['Nino3.4-T'] = xs.pearson_r(seasvars['DJF']['n34'],seasvars['DJF']['tasanom'],dim='time')
    metrics['ELI-T']     = xs.pearson_r(seasvars['DJF']['eli'],seasvars['DJF']['tasanom'],dim='time')
    metrics['Nino3.4-P'] = xs.pearson_r(seasvars['DJF']['n34'],seasvars['DJF']['pranom'],dim='time')
    metrics['ELI-P']     = xs.pearson_r(seasvars['DJF']['eli'],seasvars['DJF']['pranom'],dim='time')

    for met in ensomets:
        metrics[met+'-r'] = np.zeros((nvar, nobs))
        for ivar, var in enumerate(variants):
            tmpmod = metrics[met].copy().sel(ens=var, drop=True)
            for iobs, obs in enumerate(obsdsetlist):
                metrics[met+'-r'][ivar,iobs] = xs.pearson_r(tmpmod, metrics_allobs[met][obs],skipna=True).data
            del tmpmod
        metrics[met+'-r'] = np.nanmean(metrics[met+'-r'],axis=1)
    

    # For seasonal average correlations and standard deviations, compute each obs against each other obs then average
        # Seasonal mean temperature and precipitation maps
    for seas in seasons:
        metrics[seas+'-T'] = datin['tas'].groupby('time.month').mean(dim='time').sel(month=seasdict[seas],drop=True).mean(dim='month')
        metrics[seas+'-P'] = datin['pr'].groupby('time.month').mean(dim='time').sel(month=seasdict[seas],drop=True).mean(dim='month')

        metrics['SpaceCorr '+seas+'-T'] = np.full((nvar,nobs),np.nan)
        metrics['SpaceCorr '+seas+'-P'] = np.full((nvar,nobs),np.nan)
        metrics['SpaceSD '+seas+'-T'] = np.full((nvar,nobs),np.nan)
        metrics['SpaceSD '+seas+'-P'] = np.full((nvar,nobs),np.nan)
        for ivar, var in enumerate(variants):
            tmpT = metrics[seas+'-T'].copy().sel(ens=var, drop=True)
            tmpP = metrics[seas+'-P'].copy().sel(ens=var, drop=True)
            for iobs, obs in enumerate(obsdsetlist):
                metrics['SpaceCorr '+seas+'-T'][ivar,iobs] = xs.pearson_r(tmpT,metrics_allobs[seas+'-T'][obs],skipna=True).data
                metrics['SpaceCorr '+seas+'-P'][ivar,iobs] = xs.pearson_r(tmpP,metrics_allobs[seas+'-P'][obs],skipna=True).data

                # Normalize by standard deviation of each dataset
                metrics['SpaceSD '+seas+'-T'][ivar,iobs] = np.nanstd(tmpT.data)/np.nanstd(metrics_allobs[seas+'-T'][obs].data)
                metrics['SpaceSD '+seas+'-P'][ivar,iobs] = np.nanstd(tmpP.data)/np.nanstd(metrics_allobs[seas+'-P'][obs].data)
            del tmpT
            del tmpP
        metrics['SpaceCorr '+seas+'-T'] = np.nanmean(metrics['SpaceCorr '+seas+'-T'], axis=1)
        metrics['SpaceCorr '+seas+'-P'] = np.nanmean(metrics['SpaceCorr '+seas+'-P'], axis=1)
        metrics['SpaceSD '+seas+'-T'] = np.nanmean(metrics['SpaceSD '+seas+'-T'], axis=1)
        metrics['SpaceSD '+seas+'-P'] = np.nanmean(metrics['SpaceSD '+seas+'-P'], axis=1)

    spacemets = ['Nino3.4-T', 'Nino3.4-P', 
                 'ELI-T', 'ELI-P', 
                 'DJF-T', 'DJF-P', 
                 'MAM-T', 'MAM-P', 
                 'JJA-T', 'JJA-P', 
                 'SON-T', 'SON-P']
    spacemet_ds = xr.Dataset(
    {
        met: xr.DataArray(
            metrics[met].values,
            dims=['ens', 'lat', 'lon'],
            coords={
                'ens': (metrics['DJF-T']['ens'].data),
                'lat': (metrics['DJF-T']['lat'].data),
                'lon': (metrics['DJF-T']['lon'].data),
            }
        )
        for met in spacemets
    }
)
    spacemet_ds.to_netcdf(f'{ncdiro}{model}.{cmip}.{dom}.metric.maps.nc')

    othermets = []
    for met in metrics_allobs.keys():
        if met not in spacemets:
            othermets.append(met)

    othermets_ds = xr.Dataset()
    for met in othermets:
        if hasattr(metrics[met], 'data') and hasattr(metrics[met], 'dims'):
            othermets_ds[met] = xr.DataArray(
                metrics[met].values,
                dims=['ens'],
                coords={
                    'ens': (metrics['DJF-T']['ens'].data),
                }
            )
        else:
            # If metrics[met][obs] is a numpy array, just stack directly
            othermets_ds[met] = xr.DataArray(
                metrics[met],
                dims=['ens'],
                coords={
                'ens': (metrics['DJF-T']['ens'].data),
                }
            )
    othermets_ds.to_netcdf(f'{ncdiro}{model}.{cmip}.{dom}.metrics.nc')

    # Save metrics to pickle file
    with open(f'{diro}{filo}', 'wb') as f:
        pickle.dump(metrics, f)
    return metrics


def seasonal_avg_vars_obs(nci,obsname,latstr,lonstr,ELI=False, TASBOOL=True):
    years = list(nci.groupby('time.year').groups)
    nyr = len(years)
    drs = {}
    seaskeys=['DJF','MAM','JJA','SON']
    for seas in seaskeys:
        drs[seas] = {}
    for iy in range(nyr-1):
        drs['DJF'][iy] = slice(str(years[iy])+'-12-01',str(years[iy+1])+'-02-28')
        drs['MAM'][iy] = slice(str(years[iy+1])+'-03-01',str(years[iy+1])+'-05-30')
        drs['JJA'][iy] = slice(str(years[iy+1])+'-06-01',str(years[iy+1])+'-08-30')
        drs['SON'][iy] = slice(str(years[iy+1])+'-09-01',str(years[iy+1])+'-11-30')

    outvars = {}
    nci['pranom'] = nci['pr'].groupby('time.month') - nci['pr'].groupby('time.month').mean(dim='time')
    if TASBOOL:
        nci['tasanom'] = nci['tas'].groupby('time.month') - nci['tas'].groupby('time.month').mean(dim='time')
        outvars['tas'] = {}
        outvars['tasanom'] = {}
    if ELI:
        outvars['eli'] = {}
    outvars['pr'] = {}
    outvars['pranom'] = {}
    outvars['n34'] = {}
    datout = {}

    for seas in seaskeys:
        outvars['n34'][seas]         = np.zeros(nyr-1)
        outvars['pr'][seas]          = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        outvars['pranom'][seas]      = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        if TASBOOL:
            outvars['tas'][seas]     = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
            outvars['tasanom'][seas] = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        if ELI:
            outvars['eli'][seas]     = np.zeros(nyr-1)

        for iy in range(nyr-1):
            outvars['n34'][seas][iy]             = nci['n34'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pranom'][seas][iy,:,:]      = nci['pranom'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pr'][seas][iy,:,:]          = nci['pr'].sel(time=drs[seas][iy]).mean(dim='time').values
            if TASBOOL:
                outvars['tas'][seas][iy,:,:]     = nci['tas'].sel(time=drs[seas][iy]).mean(dim='time').values
                outvars['tasanom'][seas][iy,:,:] = nci['tasanom'].sel(time=drs[seas][iy]).mean(dim='time').values
            if ELI:
                outvars['eli'][seas][iy]         = nci['eli'].sel(time=drs[seas][iy]).mean(dim='time').values
        
        if nci[latstr].ndim == 1:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['time'], outvars['n34'][seas]),
                    pr=(['time','lat','lon'],outvars['pr'][seas]),
                    pranom=(['time','lat','lon'],outvars['pranom'][seas]),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    lat = (['lat'],nci[latstr].data),
                    lon = (['lon'],nci[lonstr].data),
                ),
                attrs=dict(description= seas + ' average variables from: ' + obsname),
            )
            if TASBOOL:
                datout[seas]['tas'] = (['time','lat','lon'],outvars['tas'][seas])
                datout[seas]['tasanom'] = (['time','lat','lon'],outvars['tasanom'][seas])
        else:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['time'], outvars['n34'][seas]),
                    pr=(['time','y','x'],outvars['pr'][seas]),
                    pranom=(['time','y','x'],outvars['pranom'][seas]),
                    lat=(['y','x'],nci['lat'].data),
                    lon=(['y','x'],nci['lon'].data),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    y = (['y'],nci['y'].data),
                    x = (['x'],nci['x'].data),
                ),
                attrs=dict(description= seas + ' average variables from: ' + obsname),
            )
            if TASBOOL:
                datout[seas]['tas'] = (['time','y','x'],outvars['tas'][seas])
                datout[seas]['tasanom'] = (['time','y','x'],outvars['tasanom'][seas])
        if ELI:
            datout[seas]['eli'] = (['time'],outvars['eli'][seas])
    return datout


def seasonal_avg_vars_model(nci,model,latstr,lonstr,ELI=False, TASBOOL=True):
    years = list(nci.groupby('time.year').groups)
    nyr = len(years)
    drs = {}
    seaskeys=['DJF','MAM','JJA','SON']
    for seas in seaskeys:
        drs[seas] = {}
    for iy in range(nyr-1):
        drs['DJF'][iy] = slice(str(years[iy])+'-12-01',str(years[iy+1])+'-02-28')
        drs['MAM'][iy] = slice(str(years[iy+1])+'-03-01',str(years[iy+1])+'-05-30')
        drs['JJA'][iy] = slice(str(years[iy+1])+'-06-01',str(years[iy+1])+'-08-30')
        drs['SON'][iy] = slice(str(years[iy+1])+'-09-01',str(years[iy+1])+'-11-30')
    nci['pranom'] = nci['pr'].groupby('time.month') - nci['pr'].groupby('time.month').mean(dim='time')
    
    datout = {}
    outvars = {}
    if TASBOOL:
        nci['tasanom'] = nci['tas'].groupby('time.month') - nci['tas'].groupby('time.month').mean(dim='time')
        outvars['tas'] = {}
        outvars['tasanom'] = {}
    if ELI:
        outvars['eli'] = {}
    outvars['pr'] = {}
    outvars['pranom'] = {}
    outvars['n34'] = {}

    ens = list(nci['ens'])
    nens = len(ens)
    nlat = len(nci[latstr])
    nlon = len(nci[lonstr])
    for seas in seaskeys:
        outvars['n34'][seas]         = np.zeros((nens, nyr-1))
        outvars['pr'][seas]          = np.zeros((nens, nyr-1,nlat,nlon))
        outvars['pranom'][seas]      = np.zeros((nens, nyr-1,nlat,nlon))
        if TASBOOL:
            outvars['tas'][seas]     = np.zeros((nens, nyr-1,nlat,nlon))
            outvars['tasanom'][seas] = np.zeros((nens, nyr-1,nlat,nlon))
        if ELI:
            outvars['eli'][seas]     = np.zeros((nens, nyr-1))

        for iy in range(nyr-1):
            outvars['n34'][seas][:,iy]             = nci['n34'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pranom'][seas][:,iy,:,:]      = nci['pranom'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pr'][seas][:,iy,:,:]          = nci['pr'].sel(time=drs[seas][iy]).mean(dim='time').values
            if TASBOOL:
                outvars['tas'][seas][:,iy,:,:]     = nci['tas'].sel(time=drs[seas][iy]).mean(dim='time').values
                outvars['tasanom'][seas][:,iy,:,:] = nci['tasanom'].sel(time=drs[seas][iy]).mean(dim='time').values
            if ELI:
                outvars['eli'][seas][:,iy]         = nci['eli'].sel(time=drs[seas][iy]).mean(dim='time').values
        
        if nci[latstr].ndim == 1:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['ens','time'], outvars['n34'][seas]),
                    pr=(['ens','time','lat','lon'],outvars['pr'][seas]),
                    pranom=(['ens','time','lat','lon'],outvars['pranom'][seas]),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    ens = (['ens'],nci['ens'].data),
                    lat = (['lat'],nci[latstr].data),
                    lon = (['lon'],nci[lonstr].data),
                ),
                attrs=dict(description= seas + ' average variables from: ' + model),
            )
            if TASBOOL:
                datout[seas]['tas'] = (['ens','time','lat','lon'],outvars['tas'][seas])
                datout[seas]['tasanom'] = (['ens','time','lat','lon'],outvars['tasanom'][seas])
        else:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['ens','time'], outvars['n34'][seas]),
                    pr=(['ens','time','y','x'],outvars['pr'][seas]),
                    pranom=(['ens','time','y','x'],outvars['pranom'][seas]),
                    lat=(['y','x'],nci['lat'].data),
                    lon=(['y','x'],nci['lon'].data),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    y = (['y'],nci['y'].data),
                    x = (['x'],nci['x'].data),
                ),
                attrs=dict(description= seas + ' average variables from: ' + model),
            )
            if TASBOOL:
                datout[seas]['tas'] = (['ens','time','y','x'],outvars['tas'][seas])
                datout[seas]['tasanom'] = (['ens','time','y','x'],outvars['tasanom'][seas])
        if ELI:
            datout[seas]['eli'] = (['ens','time'],outvars['eli'][seas])
    return datout