import glob
import xarray as xr
import xesmf
import funcs_general as fg
import numpy as np
import pandas as pd


def get_ds_metavars(imod, imeth):
    models = ['ACCESS-CM2','BCC-CSM2-MR','CanESM5','CNRM-ESM2-1','MPI-ESM1-2-HR','MRI-ESM2-0','NorESM2-MM']
    methods = ['ICAR','GARD','STAR-ESDM','LOCA2','NASA-NEX','DBCCA','DeepSD','RegCM','UCLA-WRF']
    variants = ['r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1']
    lilmodels = ['access13','canesm','cesm','miroc5','mri','noresm']
    cvdp_vars = ['_0','_0','_0','_0','_0','_0']
    return models[imod],methods[imeth],variants[imod],lilmodels[imod],cvdp_vars[imod]


def check_modmeth_cmip6(mod, meth, reg=None):
    if (meth == 'ICAR') and (mod not in ['CanESM2','NorESM2-MM']):
        CHECK=False
    elif (meth == 'DBCCA') and (mod not in ['MRI-ESM2-0','NorESM2-MM']):
        CHECK=False
    else:
        if reg:
            if meth in ['ICAR', 'GARDwest']:
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
    dvars = ['t_mean','t_range','elevation','t_max','t_min']
    gridtar = xr.open_dataset('/glade/campaign/ral/hap/trude/conus_icar/qm_data/access13_hist_exl_conv.nc',engine='netcdf4',drop_variables=dvars).isel(time=[0])
    landmask = xr.DataArray(np.tile(gridtar['pcp'], (len(dspr['time']), 1, 1)),
                 dims=['time', 'lat', 'lon'],
                 coords={'time': dspr['time'], 'lat': gridtar['lat'], 'lon': gridtar['lon']})
    dspr['lat'] = gridtar['lat']
    dspr['lon'] = gridtar['lon']
    dspr['tas'] = xr.where(~np.isnan(landmask), dspr['tas'], np.nan)
    dspr['pr']  = xr.where(~np.isnan(landmask), dspr['pr'], np.nan)
    return dspr


def read_icar_cmip6(timslic, mod, scenario='hist'):
    dvars = ['Wind']
    diri = '/glade/derecho/scratch/bkruyt/CMIP6/WUS_icar_LivGrd/'
    if scenario == 'hist':
        filis = sorted(glob.glob(mod + '_hist/daily/icar_daily*.nc'))

    dspr = xr.open_mfdataset(filis, drop_variables=dvars, engine='netcdf4').sel(time=timslic)
    dspr['tas'] = (dspr['Tmin']+dspr['Tmax'])/2
    dspr = dspr.rename({'precipitation':'Prec'})
    return dspr


def read_staresdm_cmip6(timslic, mod, variant, scenario='hist'):
    if scenario in ['hist', 'ssp245']:
        diri = '/glade/campaign/ral/hap/common/STAR-ESDM/ssp245/' + mod + '/'
    elif scenario == 'ssp585':
        diri = '/glade/campaign/ral/hap/common/STAR-ESDM/ssp585/' + mod + '/'
    else:
        print('Data does not exist for scenario: ' + scenario + ' for model: ' + mod)
        return

    start_year = pd.to_datetime(timslic.start).year
    end_year = pd.to_datetime(timslic.stop).year

    prfilis = sorted(glob.glob(diri + 'pr/downscaled*' + variant + '*.nc'))
    tasmaxfilis = sorted(glob.glob(diri + 'tasmax/downscaled*' + variant + '*.nc'))
    tasminfilis = sorted(glob.glob(diri + 'tasmin/downscaled*' + variant + '*.nc'))

    if scenario == 'hist':
        exclusion = np.arange(2015,2101)
    elif scenario in ['ssp245', 'ssp585']:
        exclusion = np.arange(1950,2015)

    for year in range(1950, 2101):
        if year in exclusion:
            prfilis.remove(diri + 'pr/downscaled.' + mod + '.' + variant + '.pr.' + scenario + 'gn.nclimgrid.star.' + str(year) + '.tllc.nc')
            tasmaxfilis.remove(diri + 'tasmax/downscaled.' + mod + '.' + variant + '.tasmax.' + scenario + 'gn.nclimgrid.star.' + str(year) + '.tllc.nc')
            tasminfilis.remove(diri + 'tasmin/downscaled.' + mod + '.' + variant + '.tasmin.' + scenario + 'gn.nclimgrid.star.' + str(year) + '.tllc.nc')
        
    dspr = xr.open_mfdataset(prfilis, engine='netcdf4').sel(time=timslic).load()
    tasmax = xr.open_mfdataset(tasmaxfilis, engine='netcdf4').sel(time=timslic).load()
    tasmin = xr.open_mfdataset(tasminfilis, engine='netcdf4').sel(time=timslic).load()
    dspr['tas'] = (tasmin['tasmin']+tasmax['tasmax'])/2
    del tasmin
    del tasmax

    return dspr


def read_LOCA_cmip6(timslic, mod, variant, scenario='hist'):
    if scenario == 'hist':
        scenario = 'historical'

    diri = '/glade/campaign/ral/hap/anewman/loca2/' + mod + '/0p0625deg/' + variant + '/' + scenario + '/'
    
    prfili = glob.glob(diri + 'pr/pr.' + mod + '.' + scenario + '.' + variant + '.1950-2014.LOCA_16thdeg_*.nc')
    dspr = xr.open_dataset(prfili).sel(time=timslic).load()
    dspr['pr'] = dspr['pr']*86400.
    
    tasmaxfili = glob.glob(diri + 'tasmax/tasmax.' + mod + '.' + scenario + '.' + variant + '.1950-2014.LOCA_16thdeg_*.nc')
    dspr['tasmax'] = xr.open_dataset(tasmaxfili)['tasmax'].sel(time=timslic).load()
    dspr['tasmax'] = dspr['tasmax']-273.15
    
    tasminfili = glob.glob(diri + 'tasmin/tasmin.' + mod + '.' + scenario + '.' + variant + '.1950-2014.LOCA_16thdeg_*.nc')
    dspr['tasmin'] = xr.open_dataset(tasminfili)['tasmin'].sel(time=timslic).load()
    dspr['tasmin'] = dspr['tasmin']-273.15
    
    dspr['tas'] = (dspr['tasmax']+dspr['tasmin'])/2
    return dspr


# def read_DBCCA_cmip6(timslic, mod, scenario='hist'):


def read_NASANEX_cmip6(timslic, mod, scenario='hist'):
    diri = '/glade/campaign/ral/hap/gutmann/downscaled_data/NASA_NEX_BCSD/'
    if scenario == 'hist':
        scenario = 'historical'
    prfilis = sorted(glob.glob(diri + 'pr/' + scenario + '/pr_day_' + mod + '_' + scenario + '_' + variant + '_*.nc'))
    tasminfilis = sorted(glob.glob(diri + 'tasmin/' + scenario + '/tasmin_day_' + mod + '_' + scenario + '_' + variant + '_*.nc'))
    tasmaxfilis = sorted(glob.glob(diri + 'tasmax/' + scenario + '/tasmax_day_' + mod + '_' + scenario + '_' + variant + '_*.nc'))\
    
    dspr = xr.open_mfdataset(prfilis).sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
    tasmin = xr.open_mfdataset(tasminfilis).sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
    tasmax = xr.open_mfdataset(tasmaxfilis).sel(time=timslic, lon=slice(230,300), lat=slice(20,55)).load()
    dspr['pr'] = dspr['pr']*86400
    dspr['tas'] = (tasmin['tasmin']+tasmax['tasmax'])/2
    dspr['tas'] = dspr['tas']-273.15
    del tasmin
    del tasmax

    dvars = ['t_mean','t_range','elevation','t_max','t_min']
    gridtar = xr.open_dataset('/glade/campaign/ral/hap/trude/conus_icar/qm_data/access13_hist_exl_conv.nc',engine='netcdf4',drop_variables=dvars)
    regr = xesmf.Regridder(dspr, gridtar, 'bilinear')
    dspr = fg.regrid_with_nan(dspr, regr)
    return dspr


def read_ds_pr_cmip6(meth, mod, timslic, lilmod, variant, scenario='hist'):
    if not check_modmeth(mod, meth):
        print('Data does not exist for this model/method combination')
        print(mod + '-' + meth)
        return

    if meth == 'ICAR':
        dspr = read_icar_cmip6(timslic, lilmod, scenario=scenario)
    elif meth == 'LOCA2':
        dspr = read_LOCA_cmip6(timslic, mod, variant, scenario=scenario)
    elif meth == 'STAR-ESDM':
        dspr = read_staresdm_cmip6(timslic, mod, variant, scenario=scenario):
    elif meth == 'NASA-NEX':
        dspr = read_NASANEX_cmip5(timslic, mod, scenario=scenario)
    elif meth == 'DBCCA':
        dspr = read_DBCCA_cmip6(timslic, mod, scenario=scenario)
    else:
        raise ValueError('Method: ' + meth + ' not valid')
        
    if dspr['lon'][0] < 0:
        dspr['lon'] = dspr['lon'] + 360.
    prstr = fg.precipname(dspr)
    dspr = dspr.rename({prstr:'pr'})
    if meth in ['GARD_r2','GARD_r3','ICAR']:
        dspr = dspr.rename({'t_mean':'tas'})
        
    if meth != 'GARDwest':
        dspr = fixnans(dspr)
    else:
        dvars = ['t_mean','t_range','elevation','t_max','t_min']
        gridtar = xr.open_dataset('/glade/campaign/ral/hap/trude/conus_icar/qm_data/access13_hist_exl_conv.nc',engine='netcdf4',drop_variables=dvars).isel(time=[0])
        landmask = xr.DataArray(np.tile(gridtar['pcp'], (len(dspr['time']), 1, 1)),
                     dims=['time', 'lat', 'lon'],
                     coords={'time': dspr['time'], 'lat': gridtar['lat'], 'lon': gridtar['lon']})
        dspr['lat'] = gridtar['lat']
        dspr['lon'] = gridtar['lon']
        dspr['pr']  = xr.where(~np.isnan(landmask), dspr['pr'], np.nan)
    return dspr
