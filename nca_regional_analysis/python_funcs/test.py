import glob
import numpy as np
import xarray as xr
import funcs_general as gen
import funcs_read_ds_cmip5 as frdc5
import icar2gmet as i2g
import xesmf
import os


def apply_forcing_to_era5(expstr):
    if expstr not in ['PGW', 'EPGW']:
        raise ValueError("expstr must be either 'PGW' or 'EPGW'")
    cesm2_varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI']

    cesm2_sfc_varlist = ['SST', 'TS',  'PS', 'PSL']
    era5_sfc_diri = '/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/'
    era5_sfc_varlist = ['128_034_sstk.ll025sc', '128_235_skt.ll025sc','128_134_sp.ll025sc', '128_151_msl.ll025sc']
    era5_sfc_name = ['SSTK', 'SKT', 'SP', 'MSL']
    era5_sfc_pids = ["34.128", "235.128", "134.128", "151.128"]

    cesm2_pl_varlist = ['T', 'U', 'V', 'Q', 'Z3', 'RELHUM']
    era5_pl_diri = '/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/'
    era5_pl_varlist = ['128_130_t.ll025sc', '128_131_u.ll025uv','128_132_v.ll025uv', '128_129_z.ll025sc', '128_157_r.ll025sc', ]
    era5_pl_name = ['T', 'U', 'V', 'Z', 'R']
    era5_pl_pids = ["130.128", "131.128", "132.128", "129.128", "157.128"]

    eventlist = ['1980_Feb', '1993_Jan', '2005_Feb', '2010_Jan']
    mon = {'1980_Feb':'02', '1993_Jan':'01', '2005_Feb':'02', '2010_Jan':'01'}
    startday= {'1980_Feb':'10', '1993_Jan':'04', '2005_Feb':'07', '2010_Jan':'16'}
    endday = {'1980_Feb':'25', '1993_Jan':'21', '2005_Feb':'16', '2010_Jan':'25'}
    
    for ivar, var in enumerate(cesm2_sfc_varlist):
        if expstr == 'EPGW':
            delta_var = load_CESM2_forcing(expstr, var)
        else:
            delta_var_Jan, delta_var_Feb = load_CESM2_forcing(expstr, var)

        for event in eventlist:
            diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
            mo = mon[event]
            if not os.path.exists(diro):
                os.makedirs(diro)

            if expstr == 'PGW':
                delta_var = pick_pgw_delta(mon[event], delta_var_Jan, delta_var_Feb)

            days = np.arange(int(startday[event]), int(endday[event])+1)
            yr = event[:4]
            for d in days:
                era5_fili = f'{yr}{mo}/e5.oper.an.sfc.{era5_sfc_varlist[ivar]}.{yr}{mo}{d}00_{yr}{mo}{d}23.nc'
                era5i = xr.open_dataset(f'{era5_sfc_diri}{era5_fili}')
                era5_filo = f'{diro}{era5_fili}'
                era5i[var] = era5i[var] + np.nan_to_num(delta_var, nan=0.0)
                era5i.to_netcdf(era5_filo)

    for ivar, var in enumerate(cesm2_pl_varlist):
        if expstr == 'EPGW':
            delta_var = load_CESM2_forcing(expstr, var)
        else:
            delta_var_Jan, delta_var_Feb = load_CESM2_forcing(expstr, var)

        for event in eventlist:
            diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
            mo = mon[event]
            if not os.path.exists(diro):
                os.makedirs(diro)

            if expstr == 'PGW':
                delta_var = pick_pgw_delta(mon[event], delta_var_Jan, delta_var_Feb)

            days = np.arange(int(startday[event]), int(endday[event])+1)
            yr = event[:4]
            for d in days:
                era5_fili = f'{yr}{mo}/e5.oper.an.pl.{era5_pl_varlist[ivar]}.{yr}{mo}{d}00_{yr}{mo}{d}23.nc'
                era5i = xr.open_dataset(f'{era5_pl_diri}{era5_fili}')
                era5_filo = f'{diro}{era5_fili}'
                era5i[var] = era5i[var] + np.nan_to_num(delta_var, nan=0.0)
                era5i.to_netcdf(era5_filo)

#### START WORKING HERE
    var= 'TSOI'


    if expstr == 'PGW':
        delta_var_Jan, delta_var_Feb = load_CESM2_forcing(expstr, var)
    else:
        delta_var = load_CESM2_forcing(expstr, var)
        delta_tsoi = {}
        delta_tsoi['STL1'] = delta_var.sel(levgrnd=slice(9.9999998e-03, 9.0000004e-02)).mean(dim='levgrnd') #0-9 cm (0-7 ERA5)
        delta_tsoi['STL2'] = delta_var.sel(levgrnd=slice(9.0000004e-02, 2.5999999e-01)).mean(dim='levgrnd') #9-26 (7-28 ERA5)
        delta_tsoi['STL3'] = delta_var.sel(levgrnd=slice(2.5999999e-01, 1.0599999e+00)).mean(dim='levgrnd') #26-100 (28-100 ERA5)
        delta_tsoi['STL4'] = delta_var.sel(levgrnd=slice( 1.0599999e+00, 2.5000000e+00)).mean(dim='levgrnd') #100-255 (100-255 ERA5)

    era5_lnd_varlist = [ '128_139_stl1.ll025sc', '128_170_stl2.ll025sc', '128_183_stl3.ll025sc', '128_236_stl4.ll025sc',]
    era5_name = ['STL1', 'STL2', 'STL3', 'STL4',]
    for event in eventlist:
        diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
        mo = mon[event]
        if not os.path.exists(diro):
            os.makedirs(diro)

        if expstr == 'PGW':
            delta_var = pick_pgw_delta(mon[event], delta_var_Jan, delta_var_Feb)
            delta_tsoi = {}
            delta_tsoi['STL1'] = delta_var.sel(levgrnd=slice(9.9999998e-03, 9.0000004e-02)).mean(dim='levgrnd') #0-9 cm (0-7 ERA5)
            delta_tsoi['STL2'] = delta_var.sel(levgrnd=slice(9.0000004e-02, 2.5999999e-01)).mean(dim='levgrnd') #9-26 (7-28 ERA5)
            delta_tsoi['STL3'] = delta_var.sel(levgrnd=slice(2.5999999e-01, 1.0599999e+00)).mean(dim='levgrnd') #26-100 (28-100 ERA5)
            delta_tsoi['STL4'] = delta_var.sel(levgrnd=slice( 1.0599999e+00, 2.5000000e+00)).mean(dim='levgrnd') #100-255 (100-255 ERA5)

        days = np.arange(int(startday[event]), int(endday[event])+1)
        yr = event[:4]
        for d in days:
            for ivar, var in enumerate(era5_name):
                era5_fili = f'{yr}{mo}/e5.oper.an.sfc.{era5_lnd_varlist[ivar]}.{yr}{mo}{d}00_{yr}{mo}{d}23.nc'
                era5i = xr.open_dataset(f'{era5_sfc_diri}{era5_fili}')
                era5_filo = f'{diro}{era5_fili}'
                era5i[var] = era5i[var] + np.nan_to_num(delta_tsoi[var], nan=0.0)
                era5i.to_netcdf(era5_filo)


def load_CESM2_forcing(expstr, var):
    diri = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/{expstr}/'
    if expstr == 'EPGW':
        hfili = f'{diri}{var}/CESM2-LE_hist_GilaRiver_events_{var}_era5grid.nc'
        ffili = f'{diri}{var}/CESM2-LE_ssp370_GilaRiver_events_{var}_era5grid.nc'
        hdat = xr.open_dataset(hfili).mean(dim=['event', 'time'])
        fdat = xr.open_dataset(ffili).mean(dim=['event', 'time'])
        delta_var = fdat[var] - hdat[var]
        return delta_var
    else:
        hfili_Jan = f'{diri}CESM2-LE_hist_GilaRiver_{var}_era5grid_Jan.nc'
        ffili_Jan = f'{diri}CESM2-LE_ssp370_GilaRiver_{var}_era5grid_Jan.nc'
        delta_var_Jan = xr.open_dataset(ffili_Jan)[var] - xr.open_dataset(hfili_Jan)[var]

        hfili_Feb = f'{diri}CESM2-LE_hist_GilaRiver_{var}_era5grid_Feb.nc'
        ffili_Feb = f'{diri}CESM2-LE_ssp370_GilaRiver_{var}_era5grid_Feb.nc'
        delta_var_Feb = xr.open_dataset(ffili_Feb)[var] - xr.open_dataset(hfili_Feb)[var]
        return delta_var_Jan, delta_var_Feb


def pick_pgw_delta(mon, delta_var_Jan, delta_var_Feb):
    if mon == '01':
        return delta_var_Jan
    elif mon == '02':
        return delta_var_Feb