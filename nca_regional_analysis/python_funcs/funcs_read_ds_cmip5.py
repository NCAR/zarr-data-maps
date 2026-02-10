import glob
import xarray as xr
import xesmf
import funcs_general as fg
import numpy as np
import pandas as pd
import cftime


def get_ds_metavars(imod=0, imeth=0, FULL_LIST=False):
    models = ['ACCESS1-3','CanESM2','CCSM4','MIROC5','MRI-CGCM3','NorESM1-M']
    # methods = ['ICAR','ICARwest','GARD_r2','GARD_r3','GARDwest','LOCA_8th','MACA','NASA-NEX']
    methods = ['ICAR','ICARwest','GARD_r2','GARD_r3','LOCA_8th','MACA','NASA-NEX']
    # methods = ['ICARv1', 'ICARv2', 'GARD-puv', 'GARD-quv', 'LOCA_8th', 'MACA', 'NASA-NEX']
    variants = ['r1i1p1','r1i1p1','r6i1p1','r1i1p1','r1i1p1','r1i1p1']
    lilmodels = ['access13','canesm','cesm','miroc5','mri','noresm']
    cvdp_vars = ['_0','_0','_5','_0','_0','_0']
    if imod >= len(models):
        return('Model index out of range')
    if imeth >= len(methods):
        return('Method index out of range')
    if FULL_LIST:
        return models,methods,variants,lilmodels,cvdp_vars
    else:
        return models[imod],methods[imeth],variants[imod],lilmodels[imod],cvdp_vars[imod]


def check_modmeth(mod, meth, reg=None):
    if (meth == 'ICARwest') and (mod in ['ACCESS1-3','NorESM1-M']):
        CHECK=False
    elif (meth == 'MACA') and (mod in ['ACCESS1-3']):
        CHECK=False
    elif (meth == 'NASA-NEX') and (mod in ['ACCESS1-3', 'CCSM4']):
        CHECK=False
    else:
        if reg:
            if meth in ['ICARwest', 'GARDwest']:
                if reg in ['Pacific Northwest', 'Pacific Southwest', 'Mountain West', 'Desert Southwest']:
                    CHECK=True
                else:
                    CHECK=False
            else:
                CHECK=True
        else:
            CHECK=True
    return CHECK


def fixnans(dspr):
    gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
    if 'time' in dspr.dims:
        landmask = xr.DataArray(np.tile(gridtar['mask'], (len(dspr['time']), 1, 1)),
                     dims=['time', 'lat', 'lon'],
                     coords={'time': dspr['time'], 'lat': gridtar['lat'], 'lon': gridtar['lon']})
    else:
        landmask = gridtar['mask']
    dspr['lat'] = gridtar['lat']
    dspr['lon'] = gridtar['lon']
    for var in dspr.data_vars:
        if var not in ['pr', 'tas', 'tasmax', 'tasmin', 't_range']:
            continue
        else:
            dspr[var] = xr.where(~np.isnan(landmask), dspr[var], np.nan)
    # dspr['tas']
    return dspr


def read_icar_cmip5(timslic, lilmod, scenario='hist'):
    dvars = ['t_range', 'elevation', 'mask']
    timstrt = timslic.start
    timend = timslic.stop
    if scenario == 'hist':
        if lilmod in ['cesm', 'ipsl', 'giss', 'miroc5', 'noresm']:
            fili1 = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/fix_{lilmod}_hist_exl_conv.nc'
            if lilmod == 'miroc5':
                fili2 = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/fix_{lilmod}_rcp85_exl_conv.nc'
            else:
                fili2 = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/fix_{lilmod}_rcp45_exl_conv.nc'
        else:
            fili1 = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/{lilmod}_hist_exl_conv.nc'
            fili2 = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/{lilmod}_rcp45_exl_conv.nc'
        # dates = pd.date_range(timstrt, timend, freq='D')
        ds1 = xr.open_dataset(fili1, drop_variables=dvars, engine='netcdf4').sel(time=slice(timstrt, '2005-12-31')).load()
        ds2 = xr.open_dataset(fili2, drop_variables=dvars, engine='netcdf4').sel(time=slice('2006-01-01', timend)).load()
        dspr = xr.concat([ds1, ds2], dim='time')

    else:
        if lilmod in ['cesm', 'ipsl', 'giss', 'miroc5', 'noresm']:
            fili = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/fix_{lilmod}_{scenario}_exl_conv.nc'
        else:
            fili = f'/glade/campaign/ral/hap/trude/conus_icar/qm_data/{lilmod}_{scenario}_exl_conv.nc'
        dspr = xr.open_dataset(fili, drop_variables=dvars, engine='netcdf4').sel(time=timslic).load()
    return dspr


def read_GARD_r2_cmip5(timslic, lilmod, scenario='hist'):
    timstrt = timslic.start
    timend = timslic.stop
    diri = '/glade/campaign/ral/hap/jhamman/storylines/GARD_downscaling_20190423/post_processed/'
    if scenario == 'hist':
        fili1 = f'{diri}gard_output.analog_regression_2.NCAR_WRF_50km.{lilmod}.hist.19510101-20051231.dm.qm.nc'
        if lilmod == 'miroc5':
            fili2 = f'{diri}gard_output.analog_regression_2.NCAR_WRF_50km.{lilmod}.rcp85.20060101-20991231.dm.qm.nc'
        else:
            fili2 = f'{diri}gard_output.analog_regression_2.NCAR_WRF_50km.{lilmod}.rcp45.20060101-20991231.dm.qm.nc'
    elif scenario == 'rcp45':
        fili = f'{diri}gard_output.analog_regression_2.NCAR_WRF_50km.{lilmod}.rcp45.20060101-20991231.dm.qm.nc'
    elif scenario == 'rcp85':
        fili = f'{diri}gard_output.analog_regression_2.NCAR_WRF_50km.{lilmod}.rcp85.20060101-20991231.dm.qm.nc'
    else:
        print('Data does not exist for GARD_r2 for the scenario: ' + scenario)
        return

    dvars = ['t_range', 'elevation', 'mask']
    if scenario == 'hist':
        ds1 = xr.open_dataset(fili1, drop_variables=dvars, engine='netcdf4').sel(time=slice(timstrt, '2005-12-31')).load()
        ds2 = xr.open_dataset(fili2, drop_variables=dvars, engine='netcdf4').sel(time=slice('2006-01-01', timend)).load()
        dspr = xr.concat([ds1, ds2], dim='time')
    else:
        dspr = xr.open_dataset(fili, drop_variables=dvars, engine='netcdf4').sel(time=timslic).load()
    return dspr


def read_GARD_r3_cmip5(timslic, lilmod, scenario='hist'):
    timstrt = timslic.start
    timend = timslic.stop
    diri = '/glade/campaign/ral/hap/jhamman/storylines/GARD_downscaling_20190423/post_processed/'
    if scenario == 'hist':
        fili1 = f'{diri}gard_output.analog_regression_3.NCAR_WRF_50km.{lilmod}.hist.19510101-20051231.dm.qm.nc'
        if lilmod == 'miroc5':
            fili2 = f'{diri}gard_output.analog_regression_3.NCAR_WRF_50km.{lilmod}.rcp85.20060101-20991231.dm.qm.nc'
        else:
            fili2 = f'{diri}gard_output.analog_regression_3.NCAR_WRF_50km.{lilmod}.rcp45.20060101-20991231.dm.qm.nc'
    elif scenario == 'rcp45':
        fili = f'{diri}gard_output.analog_regression_3.NCAR_WRF_50km.{lilmod}.rcp45.20060101-20991231.dm.qm.nc'
    elif scenario == 'rcp85':
        fili = f'{diri}gard_output.analog_regression_3.NCAR_WRF_50km.{lilmod}.rcp85.20060101-20991231.dm.qm.nc'
    else:
        print('Data does not exist for GARD_r3 for the scenario: ' + scenario)
        return
    dvars = ['t_range', 'elevation', 'mask']
    if scenario == 'hist':
        ds1 = xr.open_dataset(fili1, drop_variables=dvars, engine='netcdf4').sel(time=slice(timstrt, '2005-12-31')).load()
        ds2 = xr.open_dataset(fili2, drop_variables=dvars, engine='netcdf4').sel(time=slice('2006-01-01', timend)).load()
        dspr = xr.concat([ds1, ds2], dim='time')
    else:
        dspr = xr.open_dataset(fili, drop_variables=dvars, engine='netcdf4').sel(time=timslic).load()
    return dspr


def read_LOCA_cmip5(timslic, mod, variant, scenario='hist'):
    if scenario == 'hist':
        scenario = 'historical'
        if mod == 'MIROC5':
            tmp_prfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/pr/*'))
            tmp_prfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp85/{variant}/pr/*'))

            tmp_tasmaxfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/tasmax/*'))
            tmp_tasmaxfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp85/{variant}/tasmax/*'))

            tmp_tasminfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/tasmin/*'))
            tmp_tasminfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp85/{variant}/tasmin/*'))
        else:
            tmp_prfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/pr/*'))
            tmp_prfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp45/{variant}/pr/*'))

            tmp_tasmaxfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/tasmax/*'))
            tmp_tasmaxfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp45/{variant}/tasmax/*'))

            tmp_tasminfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/historical/{variant}/tasmin/*'))
            tmp_tasminfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/rcp45/{variant}/tasmin/*'))
        prfilis1 = []
        prfilis2 = []
        tasmaxfilis1 = []
        tasmaxfilis2 = []
        tasminfilis1 = []
        tasminfilis2 = []
        for year in range(int(timslic.start[:4]), int(timslic.stop[:4])+1):
            prfilis1.extend([f for f in tmp_prfilis1 if f'_{year}0101-' in f])
            tasmaxfilis1.extend([f for f in tmp_tasmaxfilis1 if f'_{year}0101-' in f])
            tasminfilis1.extend([f for f in tmp_tasminfilis1 if f'_{year}0101-' in f])
            prfilis2.extend([f for f in tmp_prfilis2 if f'_{year}0101-' in f])
            tasmaxfilis2.extend([f for f in tmp_tasmaxfilis2 if f'_{year}0101-' in f])
            tasminfilis2.extend([f for f in tmp_tasminfilis2 if f'_{year}0101-' in f])
        ds1 = xr.open_mfdataset(prfilis1, combine='by_coords', engine='netcdf4').sel(time=slice(timslic.start, '2005-12-31')).load()
        ds2 = xr.open_mfdataset(prfilis2, combine='by_coords', engine='netcdf4').sel(time=slice('2006-01-01', timslic.stop)).load()
        dstmx1 = xr.open_mfdataset(tasmaxfilis1, combine='by_coords', engine='netcdf4')['tasmax'].sel(time=slice(timslic.start, '2005-12-31')).load()
        dstmx2 = xr.open_mfdataset(tasmaxfilis2, combine='by_coords', engine='netcdf4')['tasmax'].sel(time=slice('2006-01-01', timslic.stop)).load()
        dstmn1 = xr.open_mfdataset(tasminfilis1, combine='by_coords', engine='netcdf4')['tasmin'].sel(time=slice(timslic.start, '2005-12-31')).load()
        dstmn2 = xr.open_mfdataset(tasminfilis2, combine='by_coords', engine='netcdf4')['tasmin'].sel(time=slice('2006-01-01', timslic.stop)).load()
        dspr = xr.concat([ds1, ds2], dim='time')
        dspr['tasmax'] = xr.concat([dstmx1, dstmx2], dim='time')-273.15
        dspr['tasmin'] = xr.concat([dstmn1, dstmn2], dim='time')-273.15
    else:
        tmp_prfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/{scenario}/{variant}/pr/*'))
        tmp_tasmaxfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/{scenario}/{variant}/tasmax/*'))
        tmp_tasminfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/common/LOCA/met/{mod}/8th/{scenario}/{variant}/tasmin/*'))
        prfilis = []
        tasmaxfilis = []
        tasminfilis = []
        for year in range(int(timslic.start[:4]), int(timslic.stop[:4])+1):
            prfilis.extend([f for f in tmp_prfilis if f'_{year}0101-' in f])
            tasmaxfilis.extend([f for f in tmp_tasmaxfilis if f'_{year}0101-' in f])
            tasminfilis.extend([f for f in tmp_tasminfilis if f'_{year}0101-' in f])
        dspr = xr.open_mfdataset(prfilis, combine='by_coords', engine='netcdf4').sel(time=timslic).load()
        dspr['tasmax'] = xr.open_mfdataset(tasmaxfilis, combine='by_coords', engine='netcdf4')['tasmax'].sel(time=timslic).load()-273.15
        dspr['tasmin'] = xr.open_mfdataset(tasminfilis, combine='by_coords', engine='netcdf4')['tasmin'].sel(time=timslic).load()-273.15
    dspr['pr'] = dspr['pr']*86400.
    dspr['tas'] = (dspr['tasmax']+dspr['tasmin'])/2
    return dspr


def read_ICARwest_cmip5(timslic, mod, scenario='hist'):
    timstrt = timslic.start
    timend = timslic.stop
    if scenario == 'hist':
        fili1 = f'/glade/work/gutmann/crb/icar_cmip5_output/{mod}_icar_hist.nc'
        if mod == 'MIROC5':
            fili2 = f'/glade/work/gutmann/crb/icar_cmip5_output/{mod}_icar_rcp85.nc'
        else:
            fili2 = f'/glade/work/gutmann/crb/icar_cmip5_output/{mod}_icar_rcp45.nc'
        ds1 = xr.open_dataset(fili1, engine='netcdf4').sel(time=slice(timstrt, '2005-12-31'))
        ds2 = xr.open_dataset(fili2, engine='netcdf4').sel(time=slice('2006-01-01', timend))
        dspr = xr.concat([ds1, ds2], dim='time').sel(time=timslic)
    else:
        fili = f'/glade/work/gutmann/crb/icar_cmip5_output/{mod}_icar_{scenario}.nc'
        dspr = xr.open_dataset(fili, engine='netcdf4').sel(time=timslic)
    dspr['tas'] = (dspr['Tmin']+dspr['Tmax'])/2
    dspr['mask'] = xr.where(~np.isnan(dspr['Wind'].isel(time=0).squeeze()), 1.0, np.nan)
    dspr = dspr.drop_vars('Wind')
    for var in ['Prec', 'Tmin', 'Tmax', 'tas']:
        tasarr = dspr[var].copy().data
        mask_expanded = np.broadcast_to(dspr['mask'].values, tasarr.shape)
        tasarr = xr.where(~np.isnan(mask_expanded), tasarr, np.nan)
        dspr[var] = (['time', 'lat', 'lon'], tasarr.data)
        del tasarr
    gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
    regr = xesmf.Regridder(dspr, gridtar, 'conservative_normed')
    dspr = regr(dspr)
    return dspr


# def read_GARDwest_cmip5(timslic, mod, scenario='hist'):
#     if scenario == 'hist':
#         filis = sorted(glob.glob('/glade/campaign/ral/hap/gutmann/gard/wus/final_output/' + mod + '/'+ mod + '19*') + 
#                    glob.glob('/glade/campaign/ral/hap/gutmann/gard/wus/final_output/' + mod + '/'+ mod + '20*'))
#         filis.pop(-1)
#     else:
#         filis = sorted(glob.glob('/glade/campaign/ral/hap/gutmann/gard/wus/final_output/' + mod + '/' + mod + '_' + scenario + '*.nc'))
#     dspr = xr.open_mfdataset(filis).sel(time=timslic).load()
#     dspr = xr.Dataset(
#         data_vars = dict(
#             pr = (['time', 'lat', 'lon'], dspr['Prec'].data),
#         ),
#         coords = dict(
#             lat = dspr['lat'][:,0].data,
#             lon = dspr['lon'][0,:].data,
#             time = dspr['time'].data
#         ),
#     )

#     gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
#     regr = xesmf.Regridder(dspr, gridtar, 'conservative')
#     dspr = fg.regrid_with_nan(dspr, regr)

#     prarr = dspr['pr'].copy().data
#     prarr[:, :42, :] = np.nan
#     prarr[:, :, 166:] = np.nan
#     dspr['pr'] = (['time', 'lat', 'lon'], prarr)
#     return dspr


def read_MACA_cmip5(timslic, mod, variant, scenario='hist', TAS=True):
    timstrt = timslic.start
    timend = timslic.stop
    if scenario == 'hist':
        scenario = 'historical'
        prfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_pr_{mod}_{variant}_historical*'))[-2:]
        if TAS:
            tasminfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmin_{mod}_{variant}_historical*'))[-2:]
            tasmaxfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmax_{mod}_{variant}_historical*'))[-2:]
        if mod == 'MIROC5':
            prfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_pr_{mod}_{variant}_rcp85*'))[0]
            if TAS:
                tasminfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmin_{mod}_{variant}_rcp85*'))[0]
                tasmaxfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmax_{mod}_{variant}_rcp85*'))[0]
        else:
            prfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_pr_{mod}_{variant}_rcp45*'))[0]
            if TAS:
                tasminfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmin_{mod}_{variant}_rcp45*'))[0]
                tasmaxfili2 = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmax_{mod}_{variant}_rcp45*'))[0]
        ds1 = xr.open_mfdataset(prfilis).sel(time=slice(timstrt, '2005-12-31')).load()
        ds2 = xr.open_dataset(prfili2).sel(time=slice('2006-01-01', timend)).load()
        dspr = xr.concat([ds1, ds2], dim='time')
        if TAS:
            dstmx1 = xr.open_mfdataset(tasmaxfilis)['air_temperature'].sel(time=slice(timstrt, '2005-12-31')).load()
            dstmn1 = xr.open_mfdataset(tasminfilis)['air_temperature'].sel(time=slice(timstrt, '2005-12-31')).load()
            dstmx2 = xr.open_dataset(tasmaxfili2)['air_temperature'].sel(time=slice('2006-01-01', timend)).load()
            dstmn2 = xr.open_dataset(tasminfili2)['air_temperature'].sel(time=slice('2006-01-01', timend)).load()
            dspr['tasmax'] = xr.concat([dstmx1, dstmx2], dim='time')-273.15
            dspr['tasmin'] = xr.concat([dstmn1, dstmn2], dim='time')-273.15
            dspr['tas'] = (dspr['tasmax']+dspr['tasmin'])/2
    else:
        prfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_pr_{mod}_{variant}_{scenario}*'))
        if TAS:
            tasminfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmin_{mod}_{variant}_{scenario}*'))
            tasmaxfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/gutmann/downscaled_data/MACA/macav2livneh_tasmax_{mod}_{variant}_{scenario}*'))
        dspr = xr.open_mfdataset(prfilis).sel(time=timslic).load()
        if TAS:
            dspr['tasmin'] = xr.open_mfdataset(tasminfilis)['air_temperature'].sel(time=timslic).load()-273.15
            dspr['tasmax'] = xr.open_mfdataset(tasmaxfilis)['air_temperature'].sel(time=timslic).load()-273.15
            dspr['tas'] = (dspr['tasmin']+dspr['tasmax'])/2

    gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
    regr = xesmf.Regridder(dspr, gridtar, 'conservative_normed')
    dspr['mask'] = xr.where(~np.isnan(dspr['precipitation'].isel(time=0).squeeze()), 1.0, np.nan)
    dspr = regr(dspr)
    return dspr


def read_NASANEX_cmip5(timslic, mod, scenario='hist'):
    if scenario == 'hist':
        scenario = 'historical'
        tmp_prfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/pr_day_BCSD_historical_r1i1p1_{mod}_*.nc'))
        tmp_tasminfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmin_day_BCSD_historical_r1i1p1_{mod}_*.nc'))
        tmp_tasmaxfilis1 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmax_day_BCSD_historical_r1i1p1_{mod}_*.nc'))
        if mod == 'MIROC5':
            tmp_prfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/pr_day_BCSD_rcp85_r1i1p1_{mod}_*.nc'))
            tmp_tasminfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmin_day_BCSD_rcp85_r1i1p1_{mod}_*.nc'))
            tmp_tasmaxfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmax_day_BCSD_rcp85_r1i1p1_{mod}_*.nc'))
        else:
            tmp_prfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/pr_day_BCSD_rcp45_r1i1p1_{mod}_*.nc'))
            tmp_tasminfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmin_day_BCSD_rcp45_r1i1p1_{mod}_*.nc'))
            tmp_tasmaxfilis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmax_day_BCSD_rcp45_r1i1p1_{mod}_*.nc'))
        prfilis1 = []
        tasminfilis1 = []
        tasmaxfilis1 = []
        prfilis2 = []
        tasminfilis2 = []
        tasmaxfilis2 = []
        for year in range(int(timslic.start[:4]), int(timslic.stop[:4])+1):
            prfilis1.extend([f for f in tmp_prfilis1 if f'_{year}' in f])
            tasminfilis1.extend([f for f in tmp_tasminfilis1 if f'_{year}' in f])
            tasmaxfilis1.extend([f for f in tmp_tasmaxfilis1 if f'_{year}' in f])
            prfilis2.extend([f for f in tmp_prfilis2 if f'_{year}' in f])
            tasminfilis2.extend([f for f in tmp_tasminfilis2 if f'_{year}' in f])
            tasmaxfilis2.extend([f for f in tmp_tasmaxfilis2 if f'_{year}' in f])
        ds1 = xr.open_mfdataset(prfilis1, engine='netcdf4').sel(time=slice(timslic.start, '2005-12-31'), lon=slice(230,300), lat=slice(20,55)).load()
        dstmn1 = xr.open_mfdataset(tasminfilis1, engine='netcdf4').sel(time=slice(timslic.start, '2005-12-31'), lon=slice(230,300), lat=slice(20,55)).load()
        dstmx1 = xr.open_mfdataset(tasmaxfilis1, engine='netcdf4').sel(time=slice(timslic.start, '2005-12-31'), lon=slice(230,300), lat=slice(20,55)).load()
        ds2 = xr.open_mfdataset(prfilis2, engine='netcdf4').sel(time=slice('2006-01-01', timslic.stop), lon=slice(230,300), lat=slice(20,55)).load()
        dstmn2 = xr.open_mfdataset(tasminfilis2, engine='netcdf4').sel(time=slice('2006-01-01', timslic.stop), lon=slice(230,300), lat=slice(20,55)).load()
        dstmx2 = xr.open_mfdataset(tasmaxfilis2, engine='netcdf4').sel(time=slice('2006-01-01', timslic.stop), lon=slice(230,300), lat=slice(20,55)).load()
        dspr = xr.concat([ds1, ds2], dim='time')
        tasmin = xr.concat([dstmn1, dstmn2], dim='time')
        tasmax = xr.concat([dstmx1, dstmx2], dim='time')
    else:
        tmp_prfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/pr_day_BCSD_{scenario}_r1i1p1_{mod}_*.nc'))
        tmp_tasminfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmin_day_BCSD_{scenario}_r1i1p1_{mod}_*.nc'))
        tmp_tasmaxfilis = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/NASA-NEX-CMIP5/{mod}/tasmax_day_BCSD_{scenario}_r1i1p1_{mod}_*.nc'))
        prfilis = []
        tasminfilis = []
        tasmaxfilis = []
        for year in range(int(timslic.start[:4]), int(timslic.stop[:4])+1):
            prfilis.extend([f for f in tmp_prfilis if f'_{year}' in f])
            tasminfilis.extend([f for f in tmp_tasminfilis if f'_{year}' in f])
            tasmaxfilis.extend([f for f in tmp_tasmaxfilis if f'_{year}' in f])
        dspr = xr.open_mfdataset(prfilis, engine='netcdf4').sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
        tasmin = xr.open_mfdataset(tasminfilis, engine='netcdf4').sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
        tasmax = xr.open_mfdataset(tasmaxfilis, engine='netcdf4').sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
    dspr['pr'] = dspr['pr']*86400
    dspr['tasmin'] = tasmin['tasmin']-273.15
    dspr['tasmax'] = tasmax['tasmax']-273.15
    dspr['tas'] = (dspr['tasmin']+dspr['tasmax'])/2
    del tasmin
    del tasmax

    gridtar = xr.open_dataset('/glade/campaign/ral/hap/nlybarger/gridtar.nc')
    regr = xesmf.Regridder(dspr, gridtar, 'bilinear')
    dspr = regr(dspr)
    return dspr


def read_GARDLENS(timslic, ens, mod='cesm2', region='CONUS'):
    diri = '/glade/campaign/collections/rda/data/d619000/'
    if mod in ['cesm2', 'canesm5']:
        datstr = '1950_2100'
    else:
        datstr = '1970_2100'
    prfili = f'{diri}pcp/GARDLENS_{mod}_{ens}_pcp_{datstr}_{region}.nc'
    dspr = xr.open_dataset(prfili).sel(time=timslic)
    tasfili = f'{diri}t_mean/GARDLENS_{mod}_{ens}_t_mean_{datstr}_{region}.nc'
    dspr['tas'] = xr.open_dataset(tasfili).sel(time=timslic)['t_mean']
    trangefili = f'{diri}t_range/GARDLENS_{mod}_{ens}_t_range_{datstr}_{region}.nc'
    dspr['t_range'] = xr.open_dataset(trangefili).sel(time=timslic)['t_range']
    dspr['tasmax'] = dspr['tas'] + dspr['t_range']/2
    dspr['tasmin'] = dspr['tas'] - dspr['t_range']/2

    return dspr


def CESM2_LE_ensemble_list(GARDLENS=False, DAILY_ALLVERT=False):
    if DAILY_ALLVERT:
        lelist = sorted(glob.glob('/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/day_1/TSOI/*BSSP370cmip6*'))
        enslist = []
        for f in lelist:
            ens = f.split('/')[-1].split('.')[4].split('-')[-1] + '.' + f.split('/')[-1].split('.')[5]
            if ens not in enslist:
                enslist.append(ens)
    else:
        lelist = sorted(glob.glob('/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/PRECT/*BHIST*'))
        enslist = []
        for f in lelist:
            ens = f.split('/')[-1].split('.')[4].split('-')[-1] + '.' + f.split('/')[-1].split('.')[5]
            if GARDLENS:
                ens = ens.replace('.0', '_')
            if ens not in enslist:
                enslist.append(ens)
    return enslist
        

def read_ds_pr_cmip5(meth, mod, timslic, lilmod, variant, scenario='hist', TAS=None):
    if mod != 'GARDLENS':
        if not check_modmeth(mod, meth):
            print('Data does not exist for this model/method combination')
            print(mod + '-' + meth)
            return

    if meth == 'ICAR':
        dspr = read_icar_cmip5(timslic, lilmod, scenario=scenario)
    elif meth == 'GARD_r2':
        dspr = read_GARD_r2_cmip5(timslic, lilmod, scenario=scenario)
    elif meth == 'GARD_r3':
        dspr = read_GARD_r3_cmip5(timslic, lilmod, scenario=scenario)
    elif meth == 'LOCA_8th':
        dspr = read_LOCA_cmip5(timslic, mod, variant, scenario=scenario)
    elif meth == 'ICARwest':
        dspr = read_ICARwest_cmip5(timslic, mod, scenario=scenario)
    # elif meth == 'GARDwest':
    #     dspr = read_GARDwest_cmip5(timslic, mod, scenario=scenario)
    elif meth == 'MACA':
        dspr = read_MACA_cmip5(timslic, mod, variant, scenario=scenario, TAS=TAS)
    elif meth == 'NASA-NEX':
        dspr = read_NASANEX_cmip5(timslic, mod, scenario=scenario)
    elif meth == 'GARDLENS':
        dspr = read_GARDLENS(timslic, variant)
    else:
        raise ValueError('Method: ' + meth + ' not valid')
        
    if dspr['lon'][0] < 0:
        dspr['lon'] = dspr['lon'] + 360.
    prstr = fg.precipname(dspr)
    dspr = dspr.rename({prstr:'pr'})
    
    if meth in ['GARD_r2','GARD_r3','ICAR']:
        dspr = dspr.rename({'t_mean':'tas', 't_max':'tasmax', 't_min':'tasmin'})
    elif meth in ['ICARwest']:
        dspr = dspr.rename({'Tmin':'tasmin','Tmax':'tasmax'})

    timstrt = str(timslic.start)
    timend = str(timslic.stop)
    no_leap = xr.date_range(start=timstrt, end=timend, freq='D', calendar='noleap', use_cftime=True)
    leap = xr.date_range(start=timstrt, end=timend, freq='D', calendar='standard', use_cftime=True)
    if len(dspr['time']) == len(no_leap):
        # no_leap calendar
        dspr = dspr.assign_coords(time=no_leap)
    elif len(dspr['time']) == len(leap):
        # leap calendar
        dspr = dspr.assign_coords(time=leap)
    else:
        print('Time dimension does not match noleap or leap year calendars')
        print(f'Data time size: {dspr.time.size}, noleap size: {no_leap.size}, leap size: {leap.size}')
        # Convert dspr['time'] to cftime.DatetimeNoLeap or cftime.DatetimeGregorian/Standard if not already

        start = dspr['time'][0].values
        end = dspr['time'][-1].values
        if isinstance(start, np.datetime64):
            start = str(start)[:10]
        else:
            start = str(start)
        if isinstance(end, np.datetime64):
            end = str(end)[:10]
        else:
            end = str(end)
        no_leap = xr.date_range(start=start, end=end, freq='D', calendar='noleap', use_cftime=True)
        leap = xr.date_range(start=start, end=end, freq='D', calendar='standard', use_cftime=True)
        if len(dspr['time']) == len(no_leap):
            dspr = dspr.assign_coords(time=no_leap)
        elif len(dspr['time']) == len(leap):
            dspr = dspr.assign_coords(time=leap)
        else:
            if len(dspr['time']) < len(no_leap):
            # Shorter than no_leap: use cftime.DatetimeNoLeap
                # dspr = dspr.assign_coords(time=[to_noleap(t) for t in dspr['time'].values])
                dspr = dspr.assign_coords(time=[to_noleap(t) for t in dspr['time'].values])
            elif len(dspr['time']) > len(no_leap):
                # Longer than no_leap: use cftime.DatetimeGregorian
                dspr = dspr.assign_coords(time=[to_gregorian(t) for t in dspr['time'].values])
            else:
                print(f'Data time size: {dspr.time.size}, noleap size: {no_leap.size}, leap size: {leap.size}')
                raise ValueError('Time dimension does not match noleap or leap year calendars')

    dspr = fixnans(dspr)
    return dspr


# def to_noleap(t):
#     if isinstance(t, cftime.DatetimeNoLeap):
#         return t
#     # If it's a numpy.datetime64 or pandas.Timestamp
#     return cftime.DatetimeNoLeap(t.year, t.month, t.day)

def to_noleap(t):
    if isinstance(t, cftime.DatetimeNoLeap):
        return t
    if isinstance(t, np.datetime64):
        t = pd.Timestamp(t).to_pydatetime()
    if isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
        return cftime.DatetimeNoLeap(t.year, t.month, t.day)
    raise TypeError(f"Cannot convert type {type(t)} to cftime.DatetimeNoLeap")

def to_gregorian(t):
    if isinstance(t, cftime.DatetimeGregorian):
        return t
    if isinstance(t, np.datetime64):
        t = pd.Timestamp(t).to_pydatetime()
    if isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
        return cftime.DatetimeGregorian(t.year, t.month, t.day)
    raise TypeError(f"Cannot convert type {type(t)} to cftime.DatetimeGregorian")
