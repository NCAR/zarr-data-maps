import xarray as xr
import glob
import numpy as np
import warnings
import pandas as pd
from funcs_read_ds_cmip5 import fixnans
import os
import icar2gmet as i2g
import xesmf
from funcs_preproc_clim_data import make_grid


def read_clim_inds(timslic):
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
    
    indy = indy.sel(time=timslic)
    return indy


def read_prism(timslic, ICG=False):
    ## END DATE: 2024-12-31
    if not ICG:
        diri = '/glade/campaign/ral/hap/common/prism/data/'
        yr0 = str(timslic)[7:11]
        yr1 = str(timslic)[21:25]
        yrlst = np.arange(int(yr0),int(yr1)+1)
        for i in range(len(yrlst)):
            if i==0:
                ncipr = xr.open_dataset(diri + 'PR/PRISM_daily_ppt_' + str(yrlst[i]) + '.nc')
                ncitas = xr.open_dataset(diri + 'T2M/PRISM_daily_tmean_' + str(yrlst[i]) + '.nc')
                ncitasmin = xr.open_dataset(diri + 'Tmin/PRISM_daily_tmin_' + str(yrlst[i]) + '.nc')
                ncitasmax = xr.open_dataset(diri + 'Tmax/PRISM_daily_tmax_' + str(yrlst[i]) + '.nc')
                ncitas['rlon'] = ncipr['rlon']
                ncitas['rlat'] = ncipr['rlat']
                ncitasmin['rlon'] = ncipr['rlon']
                ncitasmin['rlat'] = ncipr['rlat']
                ncitasmax['rlon'] = ncipr['rlon']
                ncitasmax['rlat'] = ncipr['rlat']
            else:
                tmp = xr.open_dataset(diri + 'PR/PRISM_daily_ppt_' + str(yrlst[i]) + '.nc')
                tmp['rlat'] = ncipr['rlat']
                tmp['rlon'] = ncipr['rlon']
                ncipr = xr.concat([ncipr, tmp], dim='time')
                del tmp
                tmp = xr.open_dataset(diri + 'T2M/PRISM_daily_tmean_' + str(yrlst[i]) + '.nc')
                tmp['rlat'] = ncipr['rlat']
                tmp['rlon'] = ncipr['rlon']
                ncitas = xr.concat([ncitas, tmp], dim='time')
                del tmp
                tmp = xr.open_dataset(diri + 'Tmin/PRISM_daily_tmin_' + str(yrlst[i]) + '.nc')
                tmp['rlat'] = ncipr['rlat']
                tmp['rlon'] = ncipr['rlon']
                ncitasmin = xr.concat([ncitasmin, tmp], dim='time')
                del tmp
                tmp = xr.open_dataset(diri + 'Tmax/PRISM_daily_tmax_' + str(yrlst[i]) + '.nc')
                tmp['rlat'] = ncipr['rlat']
                tmp['rlon'] = ncipr['rlon']
                ncitasmax = xr.concat([ncitasmax, tmp], dim='time')
                del tmp
        ncipr = ncipr.sel(time=timslic)
        ncipr = ncipr.reindex(rlat=list(reversed(ncipr.rlat)))
        ncitas = ncitas.sel(time=timslic)
        ncitas = ncitas.reindex(rlat=list(reversed(ncitas.rlat)))
        ncitasmin = ncitasmin.sel(time=timslic)
        ncitasmin = ncitasmin.reindex(rlat=list(reversed(ncitasmin.rlat)))
        ncitasmax = ncitasmax.sel(time=timslic)
        ncitasmax = ncitasmax.reindex(rlat=list(reversed(ncitasmax.rlat)))

        obs = xr.Dataset(
            data_vars = dict(
                pr = (['time', 'lat', 'lon'], ncipr['PR'].data),
                tas = (['time', 'lat', 'lon'], ncitas['T2M'].data),
                tasmin = (['time', 'lat', 'lon'], ncitasmin['Tmin'].data),
                tasmax = (['time', 'lat', 'lon'], ncitasmax['Tmax'].data),
            ),
            coords = dict(
                lat = ncipr['rlat'].data,
                lon = ncipr['rlon'].data+360.,
                time = ncipr['time'].data
            ),
        )
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/PRISM/'
        filis = sorted(glob.glob(diri + 'PRISM.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
        # fili = 'prism.daily.pr.tas.icargrid.nc'
        # obs = xr.open_dataset(diri + fili).sel(time=timslic).load()
    return obs


def read_conus404(timslic, ICG=False):
    ## END DATE: 2018-12-31
    if not ICG:
        # tmp = xr.open_dataset('/glade/campaign/collections/rda/data/ds559.0/INVARIANT/wrfconstants_usgs404.nc')['LANDMASK'][0,:,:]
        diri = '/glade/campaign/ral/hap/nlybarger/CONUS404_daily/'
        filis = sorted(glob.glob(diri + 'conus404.daily.pr.tas.*.nc'))
        obs = xr.open_mfdataset(filis).sel(time=timslic).load()
        # landmask = xr.DataArray(np.tile(tmp, (len(obs['time']), 1, 1)),
        #                  dims=['time', 'y', 'x'],
        #                  coords={'time': obs['time'], 'y': obs['y'], 'x': obs['x']})
        # obs['mask'] = landmask
        # obs['tas'] = xr.where(landmask==1, obs['tas'], np.nan)
        # obs['tasmin'] = xr.where(landmask==1, obs['tasmin'], np.nan)
        # obs['tasmax'] = xr.where(landmask==1, obs['tasmax'], np.nan)
        # obs['pr'] = xr.where(landmask==1, obs['pr'], np.nan)
        obs['lat'] = obs['lat'][0,:,:]
        obs['lon'] = obs['lon'][0,:,:]
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/CONUS404/'
        filis = sorted(glob.glob(diri + 'CONUS404.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_livneh(timslic, ICG=False):
    ## END DATE: 2018-12-31
    if not ICG:
        diri = '/glade/campaign/ral/hap/common/Livneh_met_updated/'
        prfilis = sorted(glob.glob(diri + 'precip/livneh_unsplit_precip.2021-05-02.*.nc'))
        ncipr = xr.open_mfdataset(prfilis).sel(Time=timslic)
        tasfilis = sorted(glob.glob(diri + 'temp_and_wind/livneh_lusu_2020_temp_and_wind.2021-05-02.*.nc'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ncitas = xr.open_mfdataset(tasfilis, drop_variables=['Wind']).sel(time=timslic)
        obs = xr.Dataset(
            data_vars = dict(
                pr = (['time', 'lat', 'lon'], ncipr['PRCP'].data),
                tas = (['time', 'lat', 'lon'], ((ncitas['Tmin']+ncitas['Tmax'])/2).data),
                tasmin = (['time', 'lat', 'lon'], ncitas['Tmin'].data),
                tasmax = (['time', 'lat', 'lon'], ncitas['Tmax'].data),
            ),
            coords = dict(
                lat = ncipr['lat'].data,
                lon = ncipr['lon'].data+360.,
                time = ncipr['Time'].data
            ),
        )
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/Livneh/'
        filis = sorted(glob.glob(diri + 'Livneh.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_nldas(timslic, ICG=False):

    ## ONLY RUNS THROUGH 2010, SHOULD DOWNLOAD UPDATED DATASET
    if not ICG:
        dvars = ['sw_avg', 'lw_avg', 'q_avg', 'pres_avg', 'uwnd_avg', 'vwnd_avg',
                 'sw_min', 'lw_min', 'q_min', 'sw_max', 'lw_max', 'q_max']
        filis = sorted(glob.glob('/glade/campaign/ral/hap/common/nldas_daily/daily_utc/*.nc'))
        filis = filis[:-2]  # remove last two files
        for i, f in enumerate(filis):
            if i==0:
                obs = xr.open_dataset(f, drop_variables=dvars)
            else:
                tmp = xr.open_dataset(f, drop_variables=dvars)
                if tmp['lon_110'].values.min() != obs['lon_110'].values.min():
                    tmp['lon_110'] = obs['lon_110']
                if tmp['lat_110'].values.min() != obs['lat_110'].values.min():
                    tmp['lat_110'] = obs['lat_110']
                obs = xr.concat([obs, tmp], dim='time')
                del tmp
        obs = obs.sel(time=timslic).load()
        obs = obs.rename({'tair_avg':'tas', 'prcp_avg':'pr', 'tair_min':'tasmin', 'tair_max':'tasmax', 'lon_110':'lon', 'lat_110':'lat'})
        obs['lon'] = obs['lon']+360.
        obs['tas'] = obs['tas']-273.15
        obs['tasmin'] = obs['tasmin']-273.15
        obs['tasmax'] = obs['tasmax']-273.15
        obs['pr'] = obs['pr']*24
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/NLDAS/'
        filis = sorted(glob.glob(diri + 'NLDAS.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_maurer(timslic, ICG=False):
    if not ICG:
        nci = xr.open_mfdataset(sorted(glob.glob('/glade/campaign/ral/hap/common/Maurer_met_full/pr/*.nc'))).sel(time=timslic).load()
        ncitas = xr.open_mfdataset(sorted(glob.glob('/glade/campaign/ral/hap/common/Maurer_met_full/tas/*.nc'))).sel(time=timslic).load()
        ncitasmin = xr.open_mfdataset(sorted(glob.glob('/glade/campaign/ral/hap/common/Maurer_met_full/tasmin/*.nc'))).sel(time=timslic).load()
        ncitasmax = xr.open_mfdataset(sorted(glob.glob('/glade/campaign/ral/hap/common/Maurer_met_full/tasmax/*.nc'))).sel(time=timslic).load()
        obs = xr.Dataset(
            data_vars = dict(
                pr = (['time', 'lat', 'lon'], nci['pr'].data),
                tas = (['time', 'lat', 'lon'], ncitas['tas'].data),
                tasmin = (['time', 'lat', 'lon'], ncitasmin['tasmin'].data),
                tasmax = (['time', 'lat', 'lon'], ncitasmax['tasmax'].data),
            ),
            coords = dict(
                lat = nci['latitude'].data,
                lon = nci['longitude'].data+360.,
                time = nci['time'].data
            ),
        )
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/Maurer/'
        filis = sorted(glob.glob(diri + 'Maurer.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_old_livneh(timslic, ICG=False):
    if not ICG:
        diri = '/glade/campaign/ral/hap/common/Livneh_met/livneh2014.1_16deg/8th/'
        filis = sorted(glob.glob(diri + 'livneh_NAmerExt_15Oct2014.*.nc'))
        nci = xr.open_mfdataset(filis, drop_variables=['wind']).sel(time=timslic)
        obs = xr.Dataset(
            data_vars = dict(
                pr = (['time', 'lat', 'lon'], nci['Prec'].data),
                tas = (['time', 'lat', 'lon'], ((nci['Tmin']+nci['Tmax'])/2).data),
                tasmin = (['time', 'lat', 'lon'], nci['Tmin'].data),
                tasmax = (['time', 'lat', 'lon'], nci['Tmax'].data),
            ),
            coords = dict(
                lat = nci['lat'].data,
                lon = nci['lon'].data+360.,
                time = nci['time'].data
            ),
        )
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/oldLivneh/'
        filis = sorted(glob.glob(diri + 'oldLivneh.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_gmet(timslic, ICG=False):
    ## END DATE: 2016-12-31
    # if not ICG:
    dvars = ['elevation']
    obs = xr.open_mfdataset(sorted(glob.glob('/glade/campaign/ral/hap/anewman/conus_v1p2/eighth/v2_landmask/*_001.nc4')), drop_variables=dvars).sel(time=timslic).load()
    obs = obs.rename({'t_mean':'tas', 'pcp':'pr'})
    obs['lon'] = obs['lon']+360.
    obs['tasmin'] = obs['tas'] - obs['t_range']/2
    obs['tasmax'] = obs['tas'] + obs['t_range']/2
    # else:
    #     diri = '/glade/campaign/ral/hap/nlybarger/OBS/GMET/'
    #     filis = sorted(glob.glob(diri + 'GMET.*.icargrid.nc'))
    #     obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_nClimGrid(timslic, ICG=False):
    ## END DATE: 2023-12-31
    if not ICG:
        dvars = []
        diri = '/glade/campaign/ral/hap/common/nClimGrid/'
        filis = sorted(glob.glob(diri + 'ncdd-*-grd-scaled.nc'))
        obs = xr.open_mfdataset(filis, drop_variables=dvars).sel(time=timslic).load()
        obs = obs.rename({'tavg':'tas', 'prcp':'pr', 'tmin':'tasmin', 'tmax':'tasmax'})
        obs['lon'] = obs['lon']+360.
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/nClimGrid/'
        filis = sorted(glob.glob(diri + 'nClimGrid.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_gridMET(timslic, ICG=False):
    ## END DATE: 2019-12-31
    if not ICG:
        diri = '/glade/campaign/ral/hap/common/gridMET/'
        prfilis = []
        tminfilis = []
        tmaxfilis = []
        for i in range(1979,2019):
            prfilis.append(f'{diri}pr_{i}.nc')
            tminfilis.append(f'{diri}tmmn_{i}.nc')
            tmaxfilis.append(f'{diri}tmmx_{i}.nc')

        ncipr = xr.open_mfdataset(prfilis, combine='by_coords').sel(day=timslic).load()
        ncitmin = xr.open_mfdataset(tminfilis, combine='by_coords').sel(day=timslic).load()
        ncitmax = xr.open_mfdataset(tmaxfilis, combine='by_coords').sel(day=timslic).load()

        obs = xr.Dataset(
            data_vars = dict(
                pr = (['time', 'lat', 'lon'], ncipr['precipitation_amount'].data),
                tas = (['time', 'lat', 'lon'], ((ncitmin['air_temperature']+ncitmax['air_temperature'])/2).data-273.15),
                tasmin = (['time', 'lat', 'lon'], ncitmin['air_temperature'].data-273.15),
                tasmax = (['time', 'lat', 'lon'], ncitmax['air_temperature'].data-273.15),
            ),
            coords = dict(
                lat = ncipr['lat'].data,
                lon = ncipr['lon'].data+360.,
                time = ncipr['day'].data
            ),
        )
    else:
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/gridMET/'
        filis = sorted(glob.glob(diri + 'gridMET.*.icargrid.nc'))
        obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs


def read_era5(timslic, grid='1deg'):
    ## END DATE: 2023-12-31
    if grid == '1deg':
        diri = '/glade/campaign/ral/hap/nlybarger/OBS/ERA5/'
        # filis = sorted(glob.glob(diri + 'era5_conus_daily_*_1deg.nc'))
        fili = 'ERA5_CONUS_daily_1981-2016_1deg.nc'
        obs = xr.open_dataset(diri+fili, engine='netcdf4').sel(time=timslic).load()
        obs = obs.rename({'T2M':'tas', 'T2Mmax':'tasmax', 'T2Mmin':'tasmin', 'PR':'pr'})
        # [['T2M', 'PR', 'T2Mmax', 'T2Mmin']].sel(time=timslic).load()
        # obs = obs.rename({'T2M':'tas', 'PR':'pr', 'T2Mmax':'tasmax', 'T2Mmin':'tasmin'})

    # elif grid == 'icargrid':
    #     diri = '/glade/campaign/ral/hap/nlybarger/OBS/ERA5/icargrid/'
    #     filis = sorted(glob.glob(diri + 'ERA5.*.icargrid.nc'))
    #     obs = xr.open_mfdataset(filis, engine='netcdf4').sel(time=timslic).load()
    return obs

def read_obs_daily(dsetname, timslic, ICARGRID=False):
    ## LIMITING FACTOR: NLDAS (2016), GMET (2016)

    if dsetname == 'PRISM':
        obs = read_prism(timslic, ICG=ICARGRID)
    elif dsetname == 'CONUS404':
        obs = read_conus404(timslic, ICG=ICARGRID)
    elif dsetname == 'Livneh':
        obs = read_livneh(timslic, ICG=ICARGRID)
    elif dsetname == 'NLDAS':
        obs = read_nldas(timslic, ICG=ICARGRID)
    elif dsetname == 'Maurer':
        obs = read_maurer(timslic, ICG=ICARGRID)
    elif dsetname == 'oldLivneh':
        obs = read_old_livneh(timslic, ICG=ICARGRID)
    elif dsetname == 'GMET':
        obs = read_gmet(timslic)#, ICG=ICARGRID)
    # elif dsetname == 'Daymet':
    elif dsetname == 'nClimGrid':
        obs = read_nClimGrid(timslic, ICG=ICARGRID)
    elif dsetname == 'gridMET':
        obs = read_gridMET(timslic, ICG=ICARGRID)
    elif dsetname == 'ERA5':
        obs = read_era5(timslic, grid='1deg')

    if obs['tas'].min() > 100:
        obs['tas'] = obs['tas']-273.15
    if obs['tasmin'].min() > 100:
        obs['tasmin'] = obs['tasmin']-273.15
    if obs['tasmax'].min() > 100:
        obs['tasmax'] = obs['tasmax']-273.15
    if ICARGRID:
        obs = fixnans(obs)
    if obs['lon'].min() < 0:
        obs['lon'] = obs['lon']+360.

    # no_leap = xr.date_range(start=timslic.start, end=timslic.stop, freq='D', calendar='noleap', use_cftime=True)
    # obs = obs.assign_coords(time=no_leap)
    return obs


def regrid_obs_icargrid(yrstrt, obsdset, OVERWRITE=True):
    # dvars = ['pcp', 't_mean', 't_range', 'elevation', 'mask', 't_max', 't_min']
    if obsdset in ['CONUS404']:
        obs_coords = read_obs_daily(obsdset, slice('1990-01-01','1990-01-01'))
        obs_coords['mask'] = xr.where(~np.isnan(obs_coords['pr']), 1., np.nan)
        gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
        gridtar = gridtar.rename_dims({'lat':'y','lon':'x'})
        latty = np.zeros((len(gridtar['y']),len(gridtar['x'])))
        lonny = np.zeros((len(gridtar['y']),len(gridtar['x'])))
        for i in range(len(gridtar['x'])):
            latty[:,i] = gridtar['lat']
        for i in range(len(gridtar['y'])):
            lonny[i,:] = gridtar['lon']
        gridtar['lat'] = (['y','x'],latty)
        gridtar['lon'] = (['y','x'],lonny)
        ds_grid_with_bounds = i2g.get_latlon_b(gridtar,lon_str='lon',lat_str='lat',lon_dim='x',lat_dim='y')
        wrf_grid_with_bounds = i2g.get_latlon_b(obs_coords,lon_str='lon',lat_str='lat',lon_dim='x',lat_dim='y')
        regridder = xesmf.Regridder(wrf_grid_with_bounds, ds_grid_with_bounds, 'conservative_normed')
    else:
        obs_coords = read_obs_daily(obsdset, slice('1990-01-01','1990-01-01'))

        obs_coords['mask'] = xr.where(~np.isnan(obs_coords['pr'].squeeze()), 1., np.nan)
        gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
        regridder = xesmf.Regridder(obs_coords, gridtar, 'conservative_normed')
    
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    diro = '/glade/campaign/ral/hap/nlybarger/OBS/' + obsdset + '/'
    nci = read_obs_daily(obsdset, slice(f'{yrstrt}-01-01',f'{yrstrt}-12-31'))
    for im in np.arange(1,13):
        if im <= 9:
            monstr = '0' + str(im)
        else:
            monstr = str(im)
        print(f'|==| Beginning to process month: {monstr}')
        filo = f'{obsdset}.daily.pr.tas.{yrstrt}-{monstr}.icargrid.nc'
        if os.path.exists(diro+filo) and not OVERWRITE:
            print(f'Daily values already computed for {yrstrt}-{monstr}')
            continue
        elif os.path.exists(diro+filo) and OVERWRITE:
            os.remove(diro+filo)
        yrstr = str(yrstrt)
        timstrt = f'{yrstr}-{monstr}-01'
        timendd = f'{yrstr}-{monstr}-{str(month_lengths[im-1])}'
        timslic = slice(timstrt, timendd)
        obs = nci.copy().sel(time=timslic)
        no_leap = xr.cftime_range(start=timslic.start, end=timslic.stop, freq='D', calendar='noleap')
        
        if obsdset not in ['CONUS404']:
            tmp = regridder(obs)
            obsout = xr.Dataset(
                data_vars = dict(
                    pr  = (['time', 'lat', 'lon'], tmp['pr'].data),
                    tas = (['time', 'lat', 'lon'], tmp['tas'].data),
                    tasmin = (['time', 'lat', 'lon'], tmp['tasmin'].data),
                    tasmax = (['time', 'lat', 'lon'], tmp['tasmax'].data),
                ),
                coords = dict(
                    time = xr.DataArray(no_leap, dims='time'),
                    lat  = (['lat'], gridtar['lat'].data),
                    lon  = (['lon'], gridtar['lon'].data),
                )
            )
        else:
            tmp = regridder(obs)
            obsout = xr.Dataset(
                data_vars = dict(
                    pr  = (['time', 'lat', 'lon'], tmp['pr'].data),
                    tas = (['time', 'lat', 'lon'], tmp['tas'].data),
                    tasmin = (['time', 'lat', 'lon'], tmp['tasmin'].data),
                    tasmax = (['time', 'lat', 'lon'], tmp['tasmax'].data),
                ),
                coords = dict(
                    time = xr.DataArray(no_leap, dims='time'),
                    lat  = (['lat'], gridtar['lat'][:,0].data),
                    lon  = (['lon'], gridtar['lon'][0,:].data),
                )
            )
        obsout = fixnans(obsout)
        # no_leap = xr.date_range(start=timslic.start, end=timslic.stop, freq='D', calendar='noleap', use_cftime=True)
        # obsout = obsout.assign_coords(time=no_leap)
        if os.path.exists(diro+filo):
            print(f'File {diro+filo} already exists. Removing it before writing new data.')
            os.remove(diro+filo)
        obsout.to_netcdf(diro+filo, mode='w')


def combine_daily_obs(obs, year=None):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    if obs == 'ERA5':
        diri = '/glade/derecho/scratch/nlybarger/era5_daily/'
        filis = sorted(glob.glob(f'{diri}era5_conus_daily_*_1deg.nc'))
        nci = xr.open_mfdataset(filis, engine='netcdf4', decode_times=time_coder).load()
        diro = '/glade/campaign/ral/hap/nlybarger/OBS/ERA5/'
        nci.to_netcdf(f'{diro}ERA5_CONUS_daily_1981-2016_1deg.nc', mode='w', format='NETCDF4')
    else:
        diri = f'/glade/campaign/ral/hap/nlybarger/OBS/{obs}/'
        # for year in range(1981, 2017):
        # for year in [2000]:
        filis = sorted(glob.glob(f'{diri}{obs}.daily.pr.tas.{year}-*.icargrid.nc'))
        nci = xr.open_mfdataset(filis, engine='netcdf4', decode_times=time_coder).load()
        diro = diri
        filo = f'{obs}.daily.pr.tas.{year}.icargrid.nc'
        if os.path.exists(f'{diro}{filo}'):
            os.remove(f'{diro}{filo}')
        nci.to_netcdf(f'{diro}{filo}', mode='w', format='NETCDF4')
