import glob
import numpy as np
import xarray as xr
import funcs_general as gen
import funcs_read_ds_cmip5 as frdc5
import icar2gmet as i2g
import xesmf
import os
import wrf


def CESM2_LE_generate_EPGW_Forcings(ensind, expind):
    xtensi = {}
    xtensi['hist'] = gen.read_pickle(f'/glade/u/home/nlybarger/scripts/PMP/xtensi_CESM2-LE_hist_GilaRiver_11.accum.pkl')
    xtensi['ssp370'] = gen.read_pickle(f'/glade/u/home/nlybarger/scripts/PMP/xtensi_CESM2-LE_ssp370_GilaRiver_11.accum.pkl')
    xtdati = {}
    xtdati['hist'] = gen.read_pickle(f'/glade/u/home/nlybarger/scripts/PMP/xtdati_CESM2-LE_hist_GilaRiver_11.accum.pkl')
    xtdati['ssp370'] = gen.read_pickle(f'/glade/u/home/nlybarger/scripts/PMP/xtdati_CESM2-LE_ssp370_GilaRiver_11.accum.pkl')

    # metvars = ['TREFHT','TREFHTMN','TREFHTMX','PRECT']
    sfc_day_varlist = ['TS','PS','PSL','PHIS', 'TREFHT', 'UBOT', 'VBOT', 'DEWPT']
    plev_day_varlist = ['T','U','V','Q','Z3','RH']
    ocn_varlist = ['SST']
    lnd_varlist = ['TSOI']
    varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI', 'TREFHT', 'UBOT', 'VBOT', 'DEWPT', 'RH']
    monlist = [1,2,3,11,12]
    # mon_varlist = ['ICEFRAC', 'SNOWHLND']

    ur_lat = 45
    ur_lon = -100 + 360
    ll_lat = 5
    ll_lon = -175 + 360
    lonslic = slice(ll_lon, ur_lon)
    latslic = slice(ll_lat, ur_lat)

    enslist = frdc5.CESM2_LE_ensemble_list(DAILY_ALLVERT=True)
    ens = enslist[ensind]

    exps = ['hist', 'ssp370']
    exp = exps[expind]  # choose 'hist' or 'ssp370'

    if exp == 'hist':
        timper = '1980-2024'
        timslic = slice('1980-01-01', '2024-12-31')
    elif exp == 'ssp370':
        timper = '2025-2069'
        timslic = slice('2025-01-01', '2069-12-31')
        
    diri = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
    ocn_diri = '/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/day_1/'
    lnd_diri = '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/day_1/'

    # varind = 0
    # for varind in range(10):
    for var in varlist:
        if var in sfc_day_varlist:
            hind = 'h1'
        elif var in plev_day_varlist:
            hind = 'h6'

        diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/EPGW/{var}/'
        filo = f'CESM2-LE_{exp}_{timper}_{var}_GilaRiver_events_{ens}.nc'
        if os.path.exists(diro + filo):
            print(f'File {diro + filo} already exists. Skipping variable {var}, ensemble {ens}, experiment {exp}.')
            continue

        if var == 'SST':
            if exp == 'hist':
                filis = sorted(glob.glob(f'{ocn_diri}{var}/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.pop.h.nday1.{var}.*.nc'))[-4:]
                filis.append(f'{ocn_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.pop.h.nday1.{var}.20150101-20250101.nc')
            else:
                filis = sorted(glob.glob(f'{ocn_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.pop.h.nday1.{var}.*.nc'))[1:6]

            ds = xr.open_mfdataset(filis,
                        combine="by_coords",
                        engine='netcdf4').sel(
                            time=timslic)[[var, 'TLONG', 'TLAT', 'time']].load()
            mask = (
                (ds['TLAT'] >= ll_lat) & (ds['TLAT'] <= ur_lat) &
                (ds['TLONG'] >= ll_lon) & (ds['TLONG'] <= ur_lon)
            )
            ds = ds.where(mask, drop=True)
        elif var == 'TSOI':
            if exp == 'hist':
                filis = sorted(glob.glob(f'{lnd_diri}{var}/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.clm2.h5.{var}.*.nc'))[-4:]
                filis.append(f'{lnd_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.clm2.h5.{var}.20150101-20241231.nc')
            elif exp == 'ssp370':
                filis = sorted(glob.glob(f'{lnd_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.clm2.h5.{var}.*.nc'))[1:6]
            ds = xr.open_mfdataset(filis,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[[var, 'time']].load()
        elif var == 'DEWPT':
            if exp == 'hist':
                filisT = sorted(glob.glob(f'{diri}TREFHT/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.TREFHT.*.nc'))[-4:]
                filisT.append(f'{diri}TREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.TREFHT.20150101-20241231.nc')
                
                filisQ = sorted(glob.glob(f'{diri}QREFHT/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.QREFHT.*.nc'))[-4:]
                filisQ.append(f'{diri}QREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.QREFHT.20150101-20241231.nc')

                filisP = sorted(glob.glob(f'{diri}PS/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.PS.*.nc'))[-4:]
                filisP.append(f'{diri}PS/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.PS.20150101-20241231.nc')
            elif exp == 'ssp370':
                filisT = sorted(glob.glob(f'{diri}TREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.TREFHT.*.nc'))[1:6]
                filisQ = sorted(glob.glob(f'{diri}QREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.QREFHT.*.nc'))[1:6]
                filisP = sorted(glob.glob(f'{diri}PS/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.PS.*.nc'))[1:6]

            dsT = xr.open_mfdataset(filisT,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[['TREFHT', 'time']].load()
            dsQ = xr.open_mfdataset(filisQ,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[['QREFHT', 'time']].load()
            dsP = xr.open_mfdataset(filisP,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[['PS', 'time']].load()
            Td = compute_dewpoint(dsT['TREFHT'], dsQ['QREFHT'], dsP['PS'])
            ds = dsT.copy()
            ds['DEWPT'] = (('time', 'lat', 'lon'), Td.values)
            ds = ds.drop_vars(['TREFHT'])
        elif var == 'RH':
            if exp == 'hist':
                filisQ = sorted(glob.glob(f'{diri}Q/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.Q.*.nc'))[-4:]
                filisQ.append(f'{diri}Q/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.Q.20150101-20241231.nc')

                filisT = sorted(glob.glob(f'{diri}T/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.T.*.nc'))[-4:]
                filisT.append(f'{diri}T/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.T.20150101-20241231.nc')

            elif exp == 'ssp370':
                filisQ = sorted(glob.glob(f'{diri}Q/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.Q.*.nc'))[1:6]
                filisT = sorted(glob.glob(f'{diri}T/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.T.*.nc'))[1:6]

            ds = xr.open_mfdataset(filisQ,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[['Q', 'time']].load()
            dsT = xr.open_mfdataset(filisT,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[['T', 'time']].load()
            ds['RH'] = gen.compute_rh_from_qt(ds['Q'], dsT['T'], ds['lev'])
            ds = ds.drop_vars(['Q'])
        else:
            if exp == 'hist':
                filis = sorted(glob.glob(f'{diri}{var}/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.*.nc'))[-4:]
                filis.append(f'{diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.20150101-20241231.nc')
            elif exp == 'ssp370':
                filis = sorted(glob.glob(f'{diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.*.nc'))[1:6]
            ds = xr.open_mfdataset(filis,
                                combine="by_coords",
                                engine='netcdf4').sel(
                                    time=timslic, 
                                    lat=latslic, 
                                    lon=lonslic)[[var, 'time']].load()
        
        # Historical period has one day of overlap due to concatenating 2015-2025 from SSP3
        if exp == 'hist':
            unique_times, counts = np.unique(ds['time'].values, return_counts=True)
            try:
                duplicate_times = unique_times[counts > 1]
                duplicate_indices = np.where(ds['time'].values == duplicate_times[0])[0]
                ds = ds.isel(time=np.delete(np.arange(ds['time'].size), duplicate_indices[0]))
            except IndexError:
                print('No duplicate times found in hist experiment.')

        print(ds[var])
        outdat = []
        eventlist = []
        for ienv in range(len(xtensi[exp])):
            if xtensi[exp][ienv] == ensind:
                tmpdat = ds.isel(time=slice(xtdati[exp][ienv] - 5, xtdati[exp][ienv] + 6))
                if tmpdat['time.month'][5].isin(monlist):
                    tmpdat = tmpdat.assign_coords(time=np.arange(-5, 6))
                    outdat.append(tmpdat[var])
                    eventlist.append(ienv)
                else:
                    continue
            else:
                continue
        if var in plev_day_varlist:
            outdata = xr.DataArray(np.stack(outdat), dims=['event', 'time', 'lev', 'lat', 'lon']).to_dataset(name=var)
        elif var == 'TSOI':
            outdata = xr.DataArray(np.stack(outdat), dims=['event', 'time', 'levgrnd', 'lat', 'lon']).to_dataset(name=var)
        else:
            outdata = xr.DataArray(np.stack(outdat), dims=['event', 'time', 'lat', 'lon']).to_dataset(name=var)
            
        outdata['event'] = eventlist
        if var != 'SST':
            outdata['lon'] = ds['lon']
            outdata['lat'] = ds['lat']
        else:
            outdata['lat'] = np.arange(ds['TLONG'].shape[0])
            outdata['lon'] = np.arange(ds['TLONG'].shape[1])
            outdata['TLONG'] = xr.DataArray(ds['TLONG'].data, dims=['lat', 'lon'])
            outdata['TLAT']  = xr.DataArray(ds['TLAT'].data,  dims=['lat', 'lon'])
            
        outdata['ens'] = xr.DataArray([ens] * len(eventlist), dims=['event'])
        outdata['date'] = xr.DataArray(ds['time'].isel(time=xtdati[exp][eventlist]).values, dims=['event'])
        outdata['time'] = xr.DataArray(np.arange(-5, 6), dims=['time'])
        outdata['lev'] = ds['lev'] if var in plev_day_varlist else None
        outdata.to_netcdf(diro + filo)
        del ds
        del outdata


def concat_regrid_CESM2_LE_EPGW_Forcings(var, exp):
    # varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI', 'TREFHT', 'UBOT', 'VBOT', 'DEWPT']
    lev_diri = '/glade/campaign/collections/gdex/data/d651056/CESM2-LE/atm/proc/tseries/month_1/T/'
    lev_fili = 'b.e21.BSSP370cmip6.f09_g17.LE2-1251.008.cam.h0.T.209501-210012.nc'
    cesm2_lev = xr.open_dataset(lev_diri + lev_fili)['lev']

    levgrnd_diri = '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/month_1/TSOI/'
    levgrnd_fili = 'b.e21.BSSP370cmip6.f09_g17.LE2-1301.010.clm2.h0.TSOI.209501-210012.nc'
    cesm2_levgrnd = xr.open_dataset(levgrnd_diri + levgrnd_fili)['levgrnd']

    plev_day_varlist = ['T','U','V','Q','Z3','RH']

    era5diri = '/glade/campaign/collections/gdex/data/d633000/e5.oper.an.pl/198001/'
    era5fili = 'e5.oper.an.pl.128_131_u.ll025uv.1980010100_1980010123.nc'
    era5 = xr.open_dataset(era5diri + era5fili)
    era5 = era5.rename({'latitude':'lat', 'longitude':'lon', 'level': 'lev'})
    # era5['lat'] = era5['lat'][::-1]
    # era5 = era5.sel(lat=slice(4,46), lon=slice(184,261))
    era5_grid_wb = i2g.get_latlon_b_rect(era5,'lon','lat','lon','lat')

    diri = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/EPGW/{var}/'
    filis = sorted(glob.glob(f'{diri}CESM2-LE_{exp}_*_GilaRiver_events_*.nc'))

    diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/EPGW/'
    filo = f'CESM2-LE_{exp}_GilaRiver_events_{var}_era5grid.nc'
    # if os.path.exists(diro + testfilo):
        # print(f'File {diro + testfilo} already exists. Skipping variable {var}, experiment {exp}.')
        # return
    for i, fili in enumerate(filis):
        if i == 0:
            ds = xr.open_dataset(fili)
        else:
            ds = xr.concat([ds, xr.open_dataset(fili)], dim='event')
    if var == 'TSOI':
        ds['levgrnd'] = cesm2_levgrnd
    ds['time'] = np.arange(-5,6)
    # Average over the 50 smallest values of event
    if ds.dims['event'] >= 50:
        smallest_events = np.argsort(ds['event'].values)[:50]
        ds = ds.isel(event=smallest_events)
    ds = ds.mean(dim=['event', 'time'])
    if var in plev_day_varlist:
        # Save original pressure levels
        ds['lev'] = cesm2_lev
        era5_lev = era5['lev'].values

        dshape = ds[var].shape
        interp_vals = np.empty((len(era5_lev), dshape[1], dshape[2]))
        for i in range(ds[var].shape[1]):  # lat
            for j in range(ds[var].shape[2]):  # lon
                interp_vals[:, i, j] = wrf.interp1d(ds[var][:,i, j], cesm2_lev, era5_lev, missing=np.nan)
        ds = xr.Dataset(
            coords={'lev': era5_lev, 'lat': ds['lat'], 'lon': ds['lon']},
            data_vars={var: (['lev', 'lat', 'lon'], interp_vals)}
        )
    if var == 'SST':
        ds['TLONG'] = ds['TLONG']
        ds['TLAT'] = ds['TLAT']
        ds_grid_wb = i2g.get_latlon_b(ds,'TLONG','TLAT','lon','lat')
    else:
        ds_grid_wb = i2g.get_latlon_b_rect(ds,'lon','lat','lon','lat')

    to_era5grid = xesmf.Regridder(ds_grid_wb, era5_grid_wb, 'bilinear')
    ds = gen.regrid_with_nan(ds,to_era5grid)
    ds.to_netcdf(diro + filo, format='NETCDF4')
    del ds


def CESM2_LE_generate_PGW_Forcings():
    # metvars = ['TREFHT','TREFHTMN','TREFHTMX','PRECT']
    sfc_day_varlist = ['TS','PS','PSL','PHIS','SST']
    plev_day_varlist = ['T','U','V','Q','Z3']
    lnd_varlist = ['TSOI']
    varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI', 'TREFHT', 'DEWPT']
    monlist = [1,2,3,11,12]
    monstrs = ['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
    # mon_varlist = ['ICEFRAC', 'SNOWHLND']

    ur_lat = 45
    ur_lon = -100 + 360
    ll_lat = 5
    ll_lon = -175 + 360
    lonslic = slice(ll_lon, ur_lon)
    latslic = slice(ll_lat, ur_lat)

    enslist = frdc5.CESM2_LE_ensemble_list(DAILY_ALLVERT=True)
    # ens = enslist[ensind]

    exps = ['hist', 'ssp370']
    # exp = exps[expind]  # choose 'hist' or 'ssp370'

        
    diri = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/month_1/'
    ocn_diri = '/glade/campaign/cgd/cesm/CESM2-LE/ocn/proc/tseries/month_1/'
    lnd_diri = '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/month_1/'

    diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/PGW/'

    for exp in exps:

        if exp == 'hist':
            timper = '1980-2024'
            timslic = slice('1980-01-01', '2024-12-31')
        elif exp == 'ssp370':
            timper = '2025-2069'
            timslic = slice('2025-01-01', '2069-12-31')

        for var in varlist:
            testfilo = f'CESM2-LE_{exp}_{timper}_{var}_GilaRiver_Dec.nc'
            if os.path.exists(diro + testfilo):
                print(f'File {diro + testfilo} already exists. Skipping variable {var}, experiment {exp}.')
                continue
            for iens, ens in enumerate(enslist):
                hind = 'h0'

                if var == 'TSOI':
                    if exp == 'hist':
                        filis = sorted(glob.glob(f'{lnd_diri}{var}/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.clm2.h0.{var}.*.nc'))[-4:]
                        filis.append(f'{lnd_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.clm2.h0.{var}.201501-202412.nc')
                    elif exp == 'ssp370':
                        filis = sorted(glob.glob(f'{lnd_diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.clm2.h0.{var}.*.nc'))[1:6]
                elif var == 'DEWPT':
                    if exp == 'hist':
                        filisT = sorted(glob.glob(f'{diri}TREFHT/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.TREFHT.*.nc'))[-4:]
                        filisT.append(f'{diri}TREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.TREFHT.201501-202412.nc')
                        
                        filisQ = sorted(glob.glob(f'{diri}QREFHT/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.QREFHT.*.nc'))[-4:]
                        filisQ.append(f'{diri}QREFHT/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.QREFHT.201501-202412.nc')

                        filisP = sorted(glob.glob(f'{diri}PS/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.PS.*.nc'))[-4:]
                        filisP.append(f'{diri}PS/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.PS.201501-202412.nc')
                    elif exp == 'ssp370':
                        filis = sorted(glob.glob(f'{diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.*.nc'))[1:6]
                else:
                    if exp == 'hist':
                        filis = sorted(glob.glob(f'{diri}{var}/b.e21.BHISTcmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.*.nc'))[-4:]
                        filis.append(f'{diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.201501-202412.nc')
                    elif exp == 'ssp370':
                        filis = sorted(glob.glob(f'{diri}{var}/b.e21.BSSP370cmip6.f09_g17.LE2-{ens}.cam.{hind}.{var}.*.nc'))[1:6]

                if var == 'DEWPT':
                    # Load TREFHT and QREFHT datasets
                    if iens == 0:
                        dsT = read_raw_CESM2_data('TREFHT', filisT, timslic, latslic, lonslic, ens)
                        dsQ = read_raw_CESM2_data('QREFHT', filisQ, timslic, latslic, lonslic, ens)
                        dsP = read_raw_CESM2_data('PS', filisP, timslic, latslic, lonslic, ens)
                    else:
                        dsT = read_raw_CESM2_data('TREFHT', filisT, timslic, latslic, lonslic, ens, dsT)
                        dsQ = read_raw_CESM2_data('QREFHT', filisQ, timslic, latslic, lonslic, ens, dsQ)
                        dsP = read_raw_CESM2_data('PS', filisP, timslic, latslic, lonslic, ens, dsP)
                else:
                    if iens == 0:
                        ds = read_raw_CESM2_data(var, filis, timslic, latslic, lonslic, ens)
                    else:
                        ds = read_raw_CESM2_data(var, filis, timslic, latslic, lonslic, ens, ds)

            if var == 'DEWPT':
                # Calculate dewpoint temperature (in Kelvin)
                # TREFHT in K, QREFHT in kg/kg, P=100000 Pa (surface)
                # Formula: Td = (243.5 * alpha) / (17.67 - alpha) + 273.15
                # where alpha = ln(RH/100) + (17.67*T)/(T+243.5)
                # But with Q and T, use:
                # RH = Q / Qs, Qs = 0.622*es/(P-0.378*es), es = 6.112*exp(17.67*T/(T+243.5))
                T = dsT['TREFHT']  # K
                Q = dsQ['QREFHT'] # kg/kg
                P = dsP['PS']  # Pa
                Td = compute_dewpoint(T, Q, P)  # K

                # Save dewpoint to ds
                ds = dsT.copy()
                ds['DEWPT'] = (('ens', 'time', 'lat', 'lon'), Td.values)
                # Drop TREFHT variable from ds if present
                ds = ds.drop_vars('TREFHT')
                del dsT
                del dsQ
                del dsP

            for imon, mon in enumerate(monlist):
                outdata = ds.sel(time=(ds['time.month'] == mon)).mean(dim=['ens', 'time'])
                outdata.to_netcdf(f'{diro}CESM2-LE_{exp}_{timper}_{var}_GilaRiver_{monstrs[imon]}.nc')
                del outdata
            del ds


def compute_dewpoint(T, Q, P):
    # T in K, Q in kg/kg, P in Pa
    T_c = T - 273.15  # Convert to Celsius
    es = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))  # hPa
    es_pa = es * 100.0  # Convert to Pa
    Qs = 0.622 * es_pa / (P - 0.378 * es_pa)
    RH = (Q / Qs) * 100.0
    RH = np.clip(RH, 0.0, 100.0)  # Limit RH to [0, 100]%
    alpha = np.log(RH / 100.0) + (17.67 * T_c) / (T_c + 243.5)
    Td = (243.5 * alpha) / (17.67 - alpha) + 273.15  # K
    return Td


def read_raw_CESM2_data(var, filis, timslic, latslic, lonslic, ens, ds=None):
    if ds is None:
        ds = xr.open_mfdataset(filis,
                        combine="by_coords",
                        engine='netcdf4').sel(
                            time=timslic, 
                            lat=latslic, 
                            lon=lonslic)[[var, 'time']].load().expand_dims(ens=[ens])
    else:
        ds = xr.concat([ds,
        xr.open_mfdataset(filis,
                            combine="by_coords",
                            engine='netcdf4').sel(
                                time=timslic, 
                                lat=latslic, 
                                lon=lonslic)[[var, 'time']].load().expand_dims(ens=[ens])],
                                dim='ens')
    return ds


def regrid_PGW_forcings(exp):
    varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI']
    plev_day_varlist = ['T','U','V','Q','Z3']
    monlist = [1,2,3,11,12]
    monstrs = ['Jan', 'Feb', 'Mar', 'Nov', 'Dec']

    if exp == 'hist':
        timper = '1980-2024'
    elif exp == 'ssp370':
        timper = '2025-2069'

    era5diri = '/glade/campaign/collections/gdex/data/d633000/e5.oper.an.pl/198001/'
    era5fili = 'e5.oper.an.pl.128_131_u.ll025uv.1980010100_1980010123.nc'
    era5 = xr.open_dataset(era5diri + era5fili)
    era5 = era5.rename({'latitude':'lat', 'longitude':'lon', 'level': 'lev'})
    # era5['lat'] = era5['lat'][::-1]
    # era5 = era5.sel(lat=slice(4,46), lon=slice(184,261))
    era5_grid_wb = i2g.get_latlon_b_rect(era5,'lon','lat','lon','lat')

    for mon in monstrs:
        for var in varlist:
            fili = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/PGW/CESM2-LE_{exp}_{timper}_{var}_GilaRiver_{mon}.nc'
            ds = xr.open_dataset(fili)

            if var in plev_day_varlist:
                # Save original pressure levels
                era5_lev = era5['lev'].values
                cesm2_lev = ds['lev'].values

                dshape = ds[var].shape
                interp_vals = np.empty((len(era5_lev), dshape[1], dshape[2]))
                for i in range(ds[var].shape[1]):  # lat
                    for j in range(ds[var].shape[2]):  # lon
                        interp_vals[:, i, j] = wrf.interp1d(ds[var][:,i, j], cesm2_lev, era5_lev, missing=np.nan)
                ds = xr.Dataset(
                    coords={'lev': era5_lev, 'lat': ds['lat'], 'lon': ds['lon']},
                    data_vars={var: (['lev', 'lat', 'lon'], interp_vals)}
                )
            ds_grid_wb = i2g.get_latlon_b_rect(ds,'lon','lat','lon','lat')
            to_era5grid = xesmf.Regridder(ds_grid_wb, era5_grid_wb, 'bilinear')
            ds = gen.regrid_with_nan(ds,to_era5grid)
            ds.to_netcdf(f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/PGW/CESM2-LE_{exp}_GilaRiver_{var}_era5grid_{mon}.nc')
            del ds


def apply_forcing_to_era5(expstr, var, OVERWRITE=False):
    if expstr not in ['PGW', 'EPGW', 'CONTROL']:
        raise ValueError("expstr must be either 'PGW' or 'EPGW'")
    cesm2_varlist = ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3', 'SST', 'TSOI', 'DEWPT']

    cesm2_sfc_varlist = ['SST', 'TS',  'PS', 'PSL', 'TREFHT', 'UBOT', 'VBOT']#, 'DEWPT'] #QREFHT
    
    era5_sfc_diri = '/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/'
    era5_sfc_varlist = ['128_034_sstk.ll025sc', '128_235_skt.ll025sc','128_134_sp.ll025sc', '128_151_msl.ll025sc'
                        ,'128_167_2t.ll025sc','128_165_10u.ll025sc','128_166_10v.ll025sc', '128_168_2d.ll025sc']
    era5_sfc_name = ['SSTK', 'SKT', 'SP', 'MSL', 'VAR_2T', 'VAR_10U', 'VAR_10V', 'VAR_2D'] #, 'RH2M'

    cesm2_pl_varlist = ['T', 'U', 'V', 'Q', 'Z3'] #, 'RELHUM']
    era5_pl_diri = '/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/'
    era5_pl_varlist = ['128_130_t.ll025sc', '128_131_u.ll025uv','128_132_v.ll025uv','128_133_q.ll025sc', '128_129_z.ll025sc'] #, '128_157_r.ll025sc', ]
    era5_pl_name = ['T', 'U', 'V', 'Q', 'Z'] #, 'R']

    # eventlist = ['1980_Feb', '1993_Jan', '2005_Feb', '2010_Jan']
    eventlist = ['1993_Jan']
    mon = {'1980_Feb':'02', '1993_Jan':'01', '2005_Feb':'02', '2010_Jan':'01'}
    startday= {'1980_Feb':'10', '1993_Jan':'04', '2005_Feb':'07', '2010_Jan':'16'}
    endday = {'1980_Feb':'25', '1993_Jan':'21', '2005_Feb':'16', '2010_Jan':'25'}
    
    # for ivar, var in enumerate(cesm2_sfc_varlist):
    if var in cesm2_sfc_varlist:
        ivar = cesm2_sfc_varlist.index(var)
        if expstr in ['EPGW', 'CONTROL']:
            delta_var = load_CESM2_forcing(expstr, var)
        else:
            delta_var_dict = load_CESM2_forcing(expstr, var)

        for event in eventlist:
            diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
            mo = mon[event]
            if not os.path.exists(diro):
                os.makedirs(diro)

            if expstr == 'PGW':
                delta_var = pick_pgw_delta(mon[event], delta_var_dict)
            
            yr = event[:4]
            monlen = gen.get_month_length(int(yr), int(mo))
            tmpdiri = f'{yr}{mo}/'
            era5_fili = f'e5.oper.an.sfc.{era5_sfc_varlist[ivar]}.{yr}{mo}0100_{yr}{mo}{monlen}23.nc'
            era5i = xr.open_dataset(f'{era5_sfc_diri}{tmpdiri}{era5_fili}')
            # era5i['latitude'] = era5i['latitude'][::-1]
            # era5i = era5i.sel(latitude=slice(4,46), longitude=slice(184,261))
            era5_filo = f'{diro}{era5_fili}'
            era5o = era5i.copy()
            era5o[era5_sfc_name[ivar]] = era5i[era5_sfc_name[ivar]] + np.nan_to_num(delta_var.data, nan=0.0)
            era5o[era5_sfc_name[ivar]].attrs = era5i[era5_sfc_name[ivar]].attrs
            era5o.to_netcdf(era5_filo)
            del era5i
            del era5o

    # for ivar, var in enumerate(cesm2_pl_varlist):
    if var in cesm2_pl_varlist:
        ivar = cesm2_pl_varlist.index(var)
        if (var == 'Q') and (expstr == 'EPGW'):
            delta_T = load_CESM2_forcing(expstr, 'T')

            for event in eventlist:
                diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
                mo = mon[event]
                if not os.path.exists(diro):
                    os.makedirs(diro)

                days = np.arange(int(startday[event]), int(endday[event])+1)
                yr = event[:4]
                tmpdiri = f'{yr}{mo}/'
                for d in days:
                    dstr = str(d).zfill(2)
                    era5_fili = f'e5.oper.an.pl.{era5_pl_varlist[ivar]}.{yr}{mo}{dstr}00_{yr}{mo}{dstr}23.nc'
                    era5_filo = f'{diro}{era5_fili}'
                    if os.path.exists(era5_filo) and not OVERWRITE:
                        continue
                    era5i = xr.open_dataset(f'{era5_pl_diri}{tmpdiri}{era5_fili}')

                    era5i['T'] = xr.open_dataset(f'{era5_pl_diri}{tmpdiri}e5.oper.an.pl.128_130_t.ll025sc.{yr}{mo}{dstr}00_{yr}{mo}{dstr}23.nc')['T']
                    era5i['T_perturbed'] = era5i['T'] + np.nan_to_num(delta_T.data, nan=0.0)

                    era5i['RH'] = gen.compute_rh_from_qt(era5i['Q'], era5i['T'], era5i['level'])
                    era5o = era5i.copy()
                    era5o = era5o.drop_vars(['Q'])
                    era5o['Q'] = gen.compute_Q_from_rh(era5i['RH'], era5i['T_perturbed'], era5i['level'])
                    era5o['Q'].attrs = era5i['Q'].attrs
                    era5o = era5o.drop_vars(['T', 'T_perturbed', 'RH'])
                    era5o = era5o.reindex(level=era5o['level'][::-1])
                    era5o.to_netcdf(era5_filo)
                    del era5i
                    del era5o
        else:
            if expstr in ['EPGW', 'CONTROL']:
                delta_var = load_CESM2_forcing(expstr, var)
            else:
                delta_var_dict = load_CESM2_forcing(expstr, var)

            for event in eventlist:
                diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
                mo = mon[event]
                if not os.path.exists(diro):
                    os.makedirs(diro)

                if expstr == 'PGW':
                    delta_var = pick_pgw_delta(mon[event], delta_var_dict)

                days = np.arange(int(startday[event]), int(endday[event])+1)
                yr = event[:4]
                tmpdiri = f'{yr}{mo}/'
                for d in days:
                    dstr = str(d).zfill(2)
                    era5_fili = f'e5.oper.an.pl.{era5_pl_varlist[ivar]}.{yr}{mo}{dstr}00_{yr}{mo}{dstr}23.nc'
                    era5_filo = f'{diro}{era5_fili}'
                    if os.path.exists(era5_filo) and not OVERWRITE:
                        continue
                    era5i = xr.open_dataset(f'{era5_pl_diri}{tmpdiri}{era5_fili}')
                    # era5i['latitude'] = era5i['latitude'][::-1]
                    # era5i = era5i.sel(latitude=slice(4,46), longitude=slice(184,261))
                    era5o = era5i.copy()
                    era5o[era5_pl_name[ivar]] = era5i[era5_pl_name[ivar]] + np.nan_to_num(delta_var.data, nan=0.0)
                    era5o[era5_pl_name[ivar]].attrs = era5i[era5_pl_name[ivar]].attrs
                    era5o = era5o.reindex(level=era5o['level'][::-1])
                    era5o.to_netcdf(era5_filo)
                    del era5i
                    del era5o


    if var == 'TSOI':
        if expstr == 'PGW':
            delta_var_dict = load_CESM2_forcing(expstr, 'TSOI')
        else:
            delta_var = load_CESM2_forcing(expstr, 'TSOI')
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
                delta_var = pick_pgw_delta(mon[event], delta_var_dict)
                delta_tsoi = {}
                delta_tsoi['STL1'] = delta_var.sel(levgrnd=slice(9.9999998e-03, 9.0000004e-02)).mean(dim='levgrnd') #0-9 cm (0-7 ERA5)
                delta_tsoi['STL2'] = delta_var.sel(levgrnd=slice(9.0000004e-02, 2.5999999e-01)).mean(dim='levgrnd') #9-26 (7-28 ERA5)
                delta_tsoi['STL3'] = delta_var.sel(levgrnd=slice(2.5999999e-01, 1.0599999e+00)).mean(dim='levgrnd') #26-100 (28-100 ERA5)
                delta_tsoi['STL4'] = delta_var.sel(levgrnd=slice( 1.0599999e+00, 2.5000000e+00)).mean(dim='levgrnd') #100-255 (100-255 ERA5)

            days = np.arange(int(startday[event]), int(endday[event])+1)
            yr = event[:4]
            monlen = gen.get_month_length(int(yr), int(mo))
            tmpdiri = f'{yr}{mo}/'
            for iv, varia in enumerate(era5_name):
                era5_fili = f'e5.oper.an.sfc.{era5_lnd_varlist[iv]}.{yr}{mo}0100_{yr}{mo}{monlen}23.nc'
                era5i = xr.open_dataset(f'{era5_sfc_diri}{tmpdiri}{era5_fili}')
                # era5i['latitude'] = era5i['latitude'][::-1]
                # era5i = era5i.sel(latitude=slice(4,46), longitude=slice(184,261))
                era5_filo = f'{diro}{era5_fili}'
                era5o = era5i.copy()
                era5o[varia] = era5i[varia] + np.nan_to_num(delta_tsoi[varia].data, nan=0.0)
                era5o[varia].attrs = era5i[varia].attrs
                era5o.to_netcdf(era5_filo)
                del era5i
                del era5o
    if var == 'DEWPT':
        if expstr in ['EPGW', 'CONTROL']:
            delta_var = load_CESM2_forcing(expstr, var)
        else:
            delta_var_dict = load_CESM2_forcing(expstr, var)

        for event in eventlist:
            diro = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/{event}/ERA5_{expstr}_files/'
            mo = mon[event]
            if not os.path.exists(diro):
                os.makedirs(diro)

            if expstr == 'PGW':
                delta_var = pick_pgw_delta(mon[event], delta_var_dict)
            
            yr = event[:4]
            monlen = gen.get_month_length(int(yr), int(mo))
            tmpdiri = f'{yr}{mo}/'
            era5_fili = f'e5.oper.an.sfc.128_168_2d.ll025sc.{yr}{mo}0100_{yr}{mo}{monlen}23.nc'
            era5i = xr.open_dataset(f'{era5_sfc_diri}{tmpdiri}{era5_fili}')
            # era5i['latitude'] = era5i['latitude'][::-1]
            # era5i = era5i.sel(latitude=slice(4,46), longitude=slice(184,261))
            era5_filo = f'{diro}{era5_fili}'
            era5o = era5i.copy()
            era5o[era5_sfc_name[ivar]] = era5i[era5_sfc_name[ivar]] + np.nan_to_num(delta_var.data, nan=0.0)
            era5o[era5_sfc_name[ivar]].attrs = era5i[era5_sfc_name[ivar]].attrs
            era5o.to_netcdf(era5_filo)
            del era5i
            del era5o


def load_CESM2_forcing(expstr, var):
    diri = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/{expstr}/'
    if expstr == 'EPGW':
        if var == 'Q':
            hfiliRH = f'{diri}CESM2-LE_hist_GilaRiver_events_RH_era5grid.nc'
            hfiliQ = f'{diri}CESM2-LE_hist_GilaRiver_events_Q_era5grid.nc'
            ffiliT = f'{diri}CESM2-LE_ssp370_GilaRiver_events_T_era5grid.nc'
            hdatRH = xr.open_dataset(hfiliRH)
            hdat = xr.open_dataset(hfiliQ)
            fdatT = xr.open_dataset(ffiliT)
            # Convert RH to specific humidity using CESM2 T and P
            fdat = gen.compute_Q_from_rh(hdatRH['RH'], fdatT['T'], hdatRH['lev'], outname='Q')
            
        else:
            hfili = f'{diri}CESM2-LE_hist_GilaRiver_events_{var}_era5grid.nc'
            ffili = f'{diri}CESM2-LE_ssp370_GilaRiver_events_{var}_era5grid.nc'
            hdat = xr.open_dataset(hfili)
            fdat = xr.open_dataset(ffili)
        if var == 'Z3':
            hdat[var] = hdat[var] * 9.81
            fdat[var] = fdat[var] * 9.81
        delta_var = fdat[var] - hdat[var]
        return delta_var
    elif expstr == 'PGW':
        delta_var_dict = {}
        for mon in ['Jan', 'Feb', 'Mar', 'Nov', 'Dec']:
            hfili = f'{diri}CESM2-LE_hist_GilaRiver_{var}_era5grid_{mon}.nc'
            ffili = f'{diri}CESM2-LE_ssp370_GilaRiver_{var}_era5grid_{mon}.nc'
            hdat = xr.open_dataset(hfili)
            fdat = xr.open_dataset(ffili)
            if var == 'Z3':
                hdat[var] = hdat[var] * 9.81
                fdat[var] = fdat[var] * 9.81
            delta_var_dict[mon] = fdat[var] - hdat[var]
        return delta_var_dict
    else:
        hfili = f'/glade/derecho/scratch/nlybarger/PMP_GilaRiver/CESM2_PGW_Forcing/EPGW/CESM2-LE_hist_GilaRiver_events_{var}_era5grid.nc'
        hdat = xr.open_dataset(hfili)
        delta_var = xr.zeros_like(hdat[var])
        return delta_var


def pick_pgw_delta(mon, delta_var_dict):
    if mon == '01':
        return delta_var_dict['Jan']
    elif mon == '02':
        return delta_var_dict['Feb']
    elif mon == '03':
        return delta_var_dict['Mar']
    elif mon == '11':
        return delta_var_dict['Nov']
    elif mon == '12':
        return delta_var_dict['Dec']
    

def generate_qg_storm_field(
    x, y, lev,
    A=1e6, x0=0, y0=0, sigma=500e3,
    f0=1e-4, g=9.81, H=8000, R=287.0, cp=1004.0
):
    """
    Generate self-consistent QG storm perturbations for ['TS', 'PS', 'PSL', 'PHIS', 'T', 'U', 'V', 'Q', 'Z3'].

    Parameters:
        x, y: 1D arrays of grid coordinates (meters)
        lev: 1D array of pressure levels (hPa)
        A: amplitude of streamfunction anomaly (m^2/s)
        x0, y0: storm center (meters)
        sigma: storm radius (meters)
        f0, g, H, R, cp: physical constants

    Returns:
        xarray.Dataset with all variables
    """
    nx, ny, nlev = len(x), len(y), len(lev)
    X, Y = np.meshgrid(x, y)
    # Baroclinic vertical structure
    def vertical_structure(p, p0=500):
        return np.sin(np.pi * (p - p[-1]) / (p0 - p[-1]))

    # QG streamfunction anomaly (ψ)
    psi = np.zeros((nlev, ny, nx))
    for k in range(nlev):
        psi[k] = A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * vertical_structure(lev[k])

    # Geostrophic wind (u, v)
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    for k in range(nlev):
        u[k] = -np.gradient(psi[k], y, axis=0)
        v[k] =  np.gradient(psi[k], x, axis=1)

    # Geopotential height anomaly (Z3)
    z3 = psi / f0

    # QG temperature anomaly (from thermal wind equation)
    T = np.zeros_like(psi)
    for k in range(1, nlev-1):
        dpsi_dp = (psi[k+1] - psi[k-1]) / (lev[k+1] - lev[k-1])
        T[k] = -f0 / R * dpsi_dp
    T[0] = T[1]
    T[-1] = T[-2]

    # QG pressure anomaly (hydrostatic balance)
    PS = np.mean(z3, axis=0) / g
    PSL = PS.copy()

    # Surface temperature (TS), surface geopotential (PHIS)
    TS = T[0]
    PHIS = z3[0]

    # Specific humidity anomaly (Q) using Clausius-Clapeyron (approximate)
    def q_sat(T, p):
        es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        return 0.622 * es / (p - es)
    Q = q_sat(T + 273.15, lev[:, None, None])

    # Package into xarray Dataset
    storm_ds = xr.Dataset(
        {
            'TS':   (['y', 'x'], TS),
            'PS':   (['y', 'x'], PS),
            'PSL':  (['y', 'x'], PSL),
            'PHIS': (['y', 'x'], PHIS),
            'T':    (['lev', 'y', 'x'], T),
            'U':    (['lev', 'y', 'x'], u),
            'V':    (['lev', 'y', 'x'], v),
            'Q':    (['lev', 'y', 'x'], Q),
            'Z3':   (['lev', 'y', 'x'], z3)
        },
        coords={
            'x': x,
            'y': y,
            'lev': lev
        }
    )
    return storm_ds