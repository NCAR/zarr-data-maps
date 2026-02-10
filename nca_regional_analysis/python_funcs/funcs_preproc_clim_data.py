import glob
import xarray as xr
import os
import numpy as np
import warnings
import funcs_general as fg
import numpy as np
import xesmf
import icar2gmet as i2g
import funcs_read_ds_cmip5 as frdc5
import pandas as pd
# import cdsapi

def ensure_time_slice_compatible(ds, time_slice):
    """
    Ensures the time_slice (a slice of strings or datetimes) matches the type of the dataset's time coordinate.
    Converts string endpoints to np.datetime64 if the time coordinate is datetime-like.
    """
    time_coord = ds["time"]
    # Only act if time_slice is a slice
    if isinstance(time_slice, slice):
        start, stop = time_slice.start, time_slice.stop
        # If time coordinate is datetime-like and endpoints are str, convert
        if np.issubdtype(time_coord.dtype, np.datetime64):
            if isinstance(start, str):
                start = np.datetime64(start)
            if isinstance(stop, str):
                stop = np.datetime64(stop)
            return slice(start, stop)
    return time_slice

def read_all_ens(nci,variable,fili,variants,exp,mip):
# This function reads all ensemble members stored for a given model
# and concatenates them into a single dataset for one variable at a time
    # Variables:
        # nci: dictionary into which the datasets are stored
        # variable: name of variable in CMIP6 file
        # fili: sorted list of CMIP6 files to be read from
        # variants: list of ensemble variant names to be read
        # exp: name of CMIP6 experiment
        # mip: CMIP5 or CMIP6
    print('|=====| Beginning to read data for ' + variable)
    print('|=====| (nens = ' + str(len(variants)) + ')')
    nfil = len(fili)
    if mip == 'CMIP6':
        if exp == 'historical':
            timslic = slice('1900-01-01','2014-12-30')
        elif exp in ['ssp245','ssp370','ssp585']:
            timslic = slice('2015-01-01','2100-12-30')
        else:
            print('Invalid experiment name')
            return
    elif mip == 'CMIP5':
        if exp == 'historical':
            timslic = slice('1900-01-01','2005-12-30')
        elif exp in ['rcp45','rcp70','rcp85']:
            timslic = slice('2006-01-01','2100-12-30')
        else:
            print('Invalid experiment name')
            return
    ie = 0
    cc = 0
    oddlatlon = False
    print(variants)
    for ifil in range(nfil):
        chckvar = variants[ie]
        if mip == 'CMIP6':
            vari = fili[ifil].split('/')[-3]
        elif mip == 'CMIP5':
            vari = fili[ifil].split('/')[-2]
        if chckvar == vari:
            if cc == 0:
                print('|=====|=====| Beginning to read in ensemble member: ' + variants[ie])
                print(fili[ifil])
                tmp = xr.open_dataset(fili[ifil],engine='netcdf4',drop_variables=['lat_bnds','lon_bnds','time_bnds','plev_bnds'])
                tmp=tmp.load()
                latstr,lonstr,latdim,londim = fg.coordnames(tmp)
                if latstr != 'lat' or lonstr != 'lon':
                    tmp = tmp.rename({lonstr: 'lon',latstr: 'lat'})
                    oglonstr = lonstr
                    oglatstr = latstr
                    lonstr = 'lon'
                    latstr = 'lat'
                    oddlatlon = True
                tmp = fg.fixlons(tmp,latdim,londim,lonstr)
                if variable == 'zg':
                    tmp = tmp.sel(plev=[50000.,20000.,],method='nearest')
                    tmp = tmp.assign_coords(plev = [50000.,20000.])
                cc += 1
            else:
                tmp2 = xr.open_dataset(fili[ifil],engine='netcdf4',drop_variables=['lat_bnds','lon_bnds','time_bnds','plev_bnds'])
                if oddlatlon:
                    tmp2 = tmp2.rename({oglonstr: 'lon',oglatstr: 'lat'})
                tmp2 = fg.fixlons(tmp2,latdim,londim,lonstr)
                if variable == 'zg':
                    tmp2 = tmp2.sel(plev=[50000.,20000.,],method='nearest')
                    tmp2 = tmp2.assign_coords(plev = [50000.,20000.])
                if tmp['lat'][0] != tmp2['lat'][0]:
                    tmp2['lat'] = tmp['lat']
                tmp = xr.concat( [tmp, tmp2], dim='time')
            if ifil != nfil-1:
                if mip == 'CMIP6':
                    if chckvar != fili[ifil+1].split('/')[-3] and ie == 0:
                        # timslic = ensure_time_slice_compatible(tmp, timslic)
                        try:
                            tmp = tmp.sel(time=timslic)
                        except KeyError:
                            print('Problem with input data for ' + variants[ie])
                            return
                        nci[variable] = tmp
                        nci[variable] = nci[variable].expand_dims(dim=dict(ens=[variants[ie]]))
                        ie += 1
                        cc = 0
                    elif chckvar != fili[ifil+1].split('/')[-3]:
                        # timslic = ensure_time_slice_compatible(tmp, timslic)
                        try:
                            tmp = tmp.sel(time=timslic)
                        except KeyError:
                            print('Problem with input data for ' + variants[ie])
                            return
                        if tmp['lat'][0] != nci[variable]['lat'][0]:
                            tmp['lat'] = nci[variable]['lat']
                        tmp = tmp.expand_dims(dim=dict(ens=[variants[ie]]))
                        nci[variable] = xr.concat( [nci[variable], tmp], dim='ens')
                        ie += 1
                        cc = 0
                elif mip == 'CMIP5':
                    if chckvar != fili[ifil+1].split('/')[-2] and ie == 0:
                        # timslic = ensure_time_slice_compatible(tmp, timslic)
                        try:
                            tmp = tmp.sel(time=timslic)
                        except KeyError:
                            print('Problem with input data for ' + variants[ie])
                            return
                        nci[variable] = tmp
                        nci[variable] = nci[variable].expand_dims(dim=dict(ens=[variants[ie]]))
                        ie += 1
                        cc = 0
                    elif chckvar != fili[ifil+1].split('/')[-2]:
                        try:
                            tmp = tmp.sel(time=timslic)
                        except KeyError:
                            print('Problem with input data for ' + variants[ie])
                            return
                        if tmp['lat'][0] != nci[variable]['lat'][0]:
                            tmp['lat'] = nci[variable]['lat']
                        tmp = tmp.expand_dims(dim=dict(ens=[variants[ie]]))
                        nci[variable] = xr.concat( [nci[variable], tmp], dim='ens')
                        ie += 1
                        cc = 0
            elif ifil == nfil-1 and ie == 0:
                print(tmp)
                print(timslic)
                try:
                    tmp = tmp.sel(time=timslic)
                except KeyError:
                    print('Problem with input data for ' + variants[ie])
                    return
                tmp = tmp.expand_dims(dim=dict(ens=[variants[ie]]))
                nci[variable] = tmp
            else:
                try:
                    tmp = tmp.sel(time=timslic)
                except KeyError:
                    print('Problem with input data for ' + variants[ie])
                    return
                if tmp['lat'][0] != nci[variable]['lat'][0]:
                    tmp['lat'] = nci[variable]['lat']
                tmp = tmp.expand_dims(dim=dict(ens=[variants[ie]]))
                nci[variable] = xr.concat( [nci[variable], tmp], dim='ens')
    print('|=====| Successfully read in data for ' + variable)
    return nci[variable]


def prep_cmip6(model,experiment):
# This function stores pr and tas CMIP6 data for a given model and experiment
# into a single netCDF file with consistent coordinate variable names
# Also computes ELI and Nino3.4 index

    if model == 'ICON-ESM-LR':
        print('|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|')
        print(model + ' has a weird grid.  Sort it out later')
        return
    elif model == 'MIROC-ES2H':
        print('|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|')
        print(model + ' data only includes 1850.  Skipping.')
        return
    
    print('|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|')
    print('Beginning preprocessing of CMIP6 data for model: ' + model + ' ' + experiment)

    diri = f'/glade/derecho/scratch/nlybarger/raw_CMIP_from_CGD/cmip6/{experiment}/Amon/'
    nci = {}
    diro = f'/glade/campaign/ral/hap/nlybarger/ESM_eval_postproc/cmip6/{experiment}/'
    filo = f'{model}.{experiment}.nc'

    if os.path.exists(diro+filo):
        print('Computation already completed for model: ' + model + ' ' + experiment)
        return
    
    if experiment == 'historical' and model == 'EC-Earth3':
        variants = ['r10i1p1f1','r11i1p1f1','r12i1p1f1','r13i1p1f1','r14i1p1f1','r15i1p1f1',
                    'r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r1i1p1f1',
                    'r21i1p1f1','r22i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1','r2i1p1f1',
                    'r3i1p1f1','r4i1p1f1','r6i1p1f1','r7i1p1f1','r9i1p1f1']
    if experiment == 'ssp245' and model == 'EC-Earth3':
        variants = ['r101i1p1f1','r102i1p1f1','r103i1p1f1','r104i1p1f1','r105i1p1f1',
                    'r106i1p1f1','r107i1p1f1','r108i1p1f1','r109i1p1f1','r10i1p1f1',
                    'r10i1p1f2','r110i1p1f1','r111i1p1f1','r112i1p1f1','r113i1p1f1',
                    'r114i1p1f1','r115i1p1f1','r116i1p1f1','r117i1p1f1','r118i1p1f1',
                    'r119i1p1f1','r11i1p1f1','r120i1p1f1','r121i1p1f1','r122i1p1f1',
                    'r123i1p1f1','r124i1p1f1','r125i1p1f1','r126i1p1f1','r127i1p1f1',
                    'r128i1p1f1','r129i1p1f1','r130i1p1f1','r131i1p1f1',
                    'r132i1p1f1','r133i1p1f1','r134i1p1f1','r135i1p1f1','r136i1p1f1',
                    'r137i1p1f1','r138i1p1f1','r139i1p1f1','r13i1p1f1','r13i1p1f2',
                    'r140i1p1f1','r141i1p1f1','r142i1p1f1','r143i1p1f1','r144i1p1f1',
                    'r145i1p1f1','r146i1p1f1','r147i1p1f1','r148i1p1f1','r149i1p1f1',
                    'r150i1p1f1','r15i1p1f1','r16i1p1f2','r18i1p1f2','r1i1p1f1','r20i1p1f2',
                    'r22i1p1f2','r24i1p1f2','r26i1p1f2','r28i1p1f2',
                    'r2i1p1f2','r4i1p1f1','r6i1p1f1','r6i1p1f2','r7i1p1f2']
    if experiment == 'ssp370' and model == 'EC-Earth3':
        variants = ['r101i1p1f1','r102i1p1f1','r103i1p1f1','r104i1p1f1','r105i1p1f1','r106i1p1f1',
                    'r107i1p1f1','r108i1p1f1','r109i1p1f1','r110i1p1f1','r111i1p1f1','r112i1p1f1','r113i1p1f1',
                    'r114i1p1f1','r115i1p1f1','r116i1p1f1','r117i1p1f1','r118i1p1f1','r119i1p1f1','r11i1p1f1',
                    'r120i1p1f1','r121i1p1f1','r122i1p1f1','r123i1p1f1','r124i1p1f1','r125i1p1f1','r126i1p1f1',
                    'r127i1p1f1','r128i1p1f1','r129i1p1f1','r130i1p1f1','r131i1p1f1','r132i1p1f1','r133i1p1f1',
                    'r134i1p1f1','r135i1p1f1','r136i1p1f1','r137i1p1f1','r138i1p1f1','r139i1p1f1','r13i1p1f1',
                    'r140i1p1f1','r141i1p1f1','r142i1p1f1','r143i1p1f1','r144i1p1f1','r145i1p1f1','r146i1p1f1',
                    'r147i1p1f1','r148i1p1f1','r149i1p1f1','r150i1p1f1','r15i1p1f1','r1i1p1f1','r4i1p1f1','r6i1p1f1','r9i1p1f1']
    if experiment == 'historical' and model == 'EC-Earth3-Veg':
        variants = ['r10i1p1f1','r12i1p1f1','r14i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1']
    if experiment == 'ssp245' and model == 'EC-Earth3-Veg':
        variants = ['r12i1p1f1','r14i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1']

    if experiment == 'ssp585' and model in ['ACCESS-CM2','ACCESS-ESM1-5','IPSL-CM6A-LR','MIROC-ES2L','MRI-ESM2-0']:
        filipr = sorted(glob.glob(diri + 'pr/' + model + '/*/*/*2015*'))
    elif experiment in ['historical','ssp245'] and model in ['EC-Earth3','EC-Earth3-Veg']:
        butt = 0
        if butt == 0:
            filipr = []
            butt = 1
        for var in variants:
            filipr.extend(sorted(glob.glob(diri + 'pr/' + model + '/' + var + '/*/*.nc')))
    else:
        filipr = sorted(glob.glob(diri + 'pr/' + model + '/*/*/*.nc'))
    
    if not filipr:
        print('pr does not exist for ' + model + ' ' + experiment)
        return
    
    # Just need to skip this code chunk for these models/experiments 
    # due to explicitly defining their ensemble members
    if experiment in ['historical','ssp245','ssp370'] and model == 'EC-Earth3':
        print('lol')
    elif experiment in ['historical','ssp245'] and model == 'EC-Earth3-Veg':
        print('lol')
    else:
        variants = ['0']
        i=0
        for ifil in range(len(filipr)):
            variant = filipr[ifil].split('/')[-3]
            if variant not in variants:
                if i > 0:
                    variants.append(variant)
                    i += 1
                else:
                    variants[i] = variant
                    i += 1
    nvar = len(variants)

    nci['pr'] = read_all_ens(nci,'pr',filipr,variants,experiment,'CMIP6')

    # Some models had superfluous files floating around, 
    # so took special exceptions to pick out the right time period
    if experiment == 'ssp585' and model in ['ACCESS-CM2','ACCESS-ESM1-5','IPSL-CM6A-LR','MIROC-ES2L','MRI-ESM2-0']:
        filitas = sorted(glob.glob(diri + 'tas/' + model + '/*/*/*2015*'))
    elif experiment in ['historical','ssp245'] and model in ['EC-Earth3','EC-Earth3-Veg']:
        butt = 0
        if butt == 0:
            filitas = []
            butt = 1
        for var in variants:
            filitas.extend(sorted(glob.glob(diri + 'tas/' + model + '/' + var + '/*/*')))
    else:
        filitas = sorted(glob.glob(diri + 'tas/' + model + '/*/*/*'))

    nci['tas'] = read_all_ens(nci,'tas',filitas,variants,experiment,'CMIP6')

    varlist = list(nci.keys())
    latstr,lonstr,latdim,londim = fg.coordnames(nci['pr'])
    
    # Compute ELI and Nino3.4 from 2-m air temperature
        # 2-m air temp has been shown to be a very good proxy for SST, and
        # is always on the same grid as the atmospheric variables, so is 
        # easier to work with
    ntim = len(nci['tas']['time'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)

        troptas = nci['tas']['tas'].sel(lat=slice(-5.,5.),drop=True).mean(dim=[latdim,londim],skipna=True)
        tmp  = nci['tas']['tas'].sel(lat=slice(-5.,5.),lon=slice(130.,275.),drop=True)
        troptasval = np.zeros((nvar,ntim,tmp.shape[2],tmp.shape[3]))
        for i in range(tmp.shape[2]):
            for j in range(tmp.shape[3]):
                troptasval[:,:,i,j] = troptas.data
        sstanom = tmp - troptasval
        print('|=====| Beginning computation of ELI')
        eli_mon = np.zeros((nvar,ntim))
        if sstanom[lonstr].ndim == 1:
            londat = np.zeros((sstanom[latstr].shape[0],sstanom[lonstr].shape[0]))
            for i in range(sstanom[latstr].shape[0]):
                londat[i,:] = sstanom[lonstr]

            lonny = xr.DataArray(
                        data = londat,
                        dims = [latdim, londim],
                        coords = dict(
                            londim = ([londim], sstanom[londim].data),
                            latdim = ([latdim], sstanom[latdim].data)),)
        else:
            lonny = sstanom[lonstr]
        for it in range(ntim):
            for iens in range(nvar):
                eli_mon[iens,it] = lonny.where(sstanom[iens,it,:,:] > 0., drop=True).mean(skipna=True)

        n34_sst = nci['tas'].sel(lat=slice(-5,5),lon=slice(190,240),drop=True)

        tmp = n34_sst.groupby('time.month') - n34_sst.groupby('time.month').mean(dim='time')
        tmp = tmp.mean(dim=[latdim,londim],skipna=True)
        n34 = tmp.rolling(time=5,center=True).mean(dim='time')
        n34['tas'][:,0] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2])/3
        n34['tas'][:,1] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3])/4
        n34['tas'][:,2] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4])/5
        n34['tas'][:,3] = (tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4] + tmp['tas'][:,5])/5
        n34['tas'][:,4] = (tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4] + tmp['tas'][:,5] + tmp['tas'][:,6])/5

        n34['tas'][:,-1] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3])/3
        n34['tas'][:,-2] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4])/4
        n34['tas'][:,-3] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5])/5
        n34['tas'][:,-4] = (tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5] + tmp['tas'][:,-6])/5
        n34['tas'][:,-5] = (tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5] + tmp['tas'][:,-6] + tmp['tas'][:,-7])/5
        print(n34)

        enso_ind = xr.Dataset(
        data_vars = dict(
                    eli=(['ens','time'], eli_mon),
                    n34=(['ens','time'], n34['tas'].data),
                    ),
                coords = dict(
                    time = nci['tas']['time'].data,
                    ens = variants,
                    ),
                attrs=dict(description='ELI and Nino 3.4 data from: '+model),
                    )
        print('|=====| Completed computation of ELI')

    nci['pr']['pr'] = nci['pr']['pr']*86400

    nco = nci['pr']['pr'].to_dataset()
    nco = nco.assign(tas=nci['tas']['tas'])
    nco = nco.assign(eli=enso_ind['eli'])
    nco = nco.assign(n34=enso_ind['n34'])

    nco.to_netcdf(diro+filo,mode='w',format='NETCDF4')
    print('Completed preprocessing of CMIP6 data for model: ' + model)

    # Explicit memory cleanup
    import gc
    del nco, nci, n34_sst, n34, troptas, troptasval, tmp, sstanom, enso_ind
    gc.collect()


def prep_cmip5(model,experiment):
    # This function stores pr and tas CMIP5 data for a given model and experiment
    # into a single netCDF file with consistent coordinate variable names
    # Also computes ELI and Nino3.4 index

    print('|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|=====|')
    print('Beginning preprocessing of CMIP5 data for model: ' + model + ' ' + experiment)

    diri = f'/glade/derecho/scratch/nlybarger/raw_CMIP_from_CGD/cmip5/{experiment}/Amon/'
    nci = {}
    diro = f'/glade/campaign/ral/hap/nlybarger/ESM_eval_postproc/cmip5/{experiment}/'
    filo = f'{model}.{experiment}.nc'

    if os.path.exists(diro+filo):
        print('Computation already completed for model: ' + model + ' ' + experiment)
        return
    
    if experiment == 'historical' and model == 'EC-EARTH':
        variants = ['r12i1p1', 'r2i1p1', 'r6i1p1', 'r8i1p1', 'r9i1p1']
    if experiment == 'historical' and model == 'GISS-E2-H':
        variants = ['r1i1p1', 'r1i1p2', 'r1i1p3', 'r2i1p1', 'r2i1p2', 
                    'r2i1p3', 'r3i1p1', 'r3i1p2', 'r3i1p3', 'r4i1p1', 
                    'r4i1p2', 'r4i1p3', 'r5i1p1', 'r5i1p2', 'r5i1p3', 
                    'r6i1p2']
    
    if experiment == 'historical' and model in ['EC-EARTH','GISS-E2-H']:
        butt = 0
        if butt == 0:
            filipr = []
            butt = 1
        for var in variants:
            filipr.extend(sorted(glob.glob(diri + 'pr/' + model + '/' + var + '/*.nc')))
    else:
        filipr = sorted(glob.glob(diri + 'pr/' + model + '/*/*.nc'))
    
    if not filipr:
        print('pr does not exist for ' + model + ' ' + experiment)
        return
    
# Just need to skip this code chunk for these models/experiments 
# due to explicitly defining their ensemble members
    if experiment == 'historical' and model in ['EC-EARTH','GISS-E2-H']:
        print('lol')
    else:
        variants = ['0']
        i=0
        for ifil in range(len(filipr)):
            variant = filipr[ifil].split('/')[-2]
            if variant not in variants:
                if i > 0:
                    variants.append(variant)
                    i += 1
                else:
                    variants[i] = variant
                    i += 1
    nvar = len(variants)

    nci['pr'] = read_all_ens(nci,'pr',filipr,variants,experiment,'CMIP5')
    if nci['pr'] is None:
        print('Failed to read pr for ' + model + ' ' + experiment)
        return
# Some models had superfluous files floating around, 
# so took special exceptions to pick out the right time period
    if experiment =='historical' and model in ['EC-EARTH','GISS-E2-H']:
        butt = 0
        if butt == 0:
            filitas = []
            butt = 1
        for var in variants:
            filitas.extend(sorted(glob.glob(diri + 'tas/' + model + '/' + var + '/*')))
    else:
        filitas = sorted(glob.glob(diri + 'tas/' + model + '/*/*'))
# Some models had superfluous files floating around, 
# so took special exceptions to pick out the right time period

    nci['tas'] = read_all_ens(nci,'tas',filitas,variants,experiment,'CMIP5')
    if nci['tas'] is None:
        print('Failed to read tas for ' + model + ' ' + experiment)
        return
    varlist = list(nci.keys())
    latstr,lonstr,latdim,londim = fg.coordnames(nci['pr'])
    
# Compute ELI and Nino3.4 from 2-m air temperature
    # 2-m air temp has been shown to be a very good proxy for SST, and
    # is always on the same grid as the atmospheric variables, so is 
    # easier to work with
    ntim = len(nci['tas']['time'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)

        troptas = nci['tas']['tas'].sel(lat=slice(-5.,5.),drop=True).mean(dim=[latdim,londim],skipna=True)
        tmp  = nci['tas']['tas'].sel(lat=slice(-5.,5.),lon=slice(130.,275.),drop=True)
        troptasval = np.zeros((nvar,ntim,tmp.shape[2],tmp.shape[3]))
        for i in range(tmp.shape[2]):
            for j in range(tmp.shape[3]):
                troptasval[:,:,i,j] = troptas.data
        sstanom = tmp - troptasval
        print('|=====| Beginning computation of ELI')
        eli_mon = np.zeros((nvar,ntim))
        if sstanom[lonstr].ndim == 1:
            londat = np.zeros((sstanom[latstr].shape[0],sstanom[lonstr].shape[0]))
            for i in range(sstanom[latstr].shape[0]):
                londat[i,:] = sstanom[lonstr]

            lonny = xr.DataArray(
                        data = londat,
                        dims = [latdim, londim],
                        coords = dict(
                            londim = ([londim], sstanom[londim].data),
                            latdim = ([latdim], sstanom[latdim].data)),)
        else:
            lonny = sstanom[lonstr]
        for it in range(ntim):
            for iens in range(nvar):
                eli_mon[iens,it] = lonny.where(sstanom[iens,it,:,:] > 0., drop=True).mean(skipna=True)

        n34_sst = nci['tas'].sel(lat=slice(-5,5),lon=slice(190,240),drop=True)

        tmp = n34_sst.groupby('time.month') - n34_sst.groupby('time.month').mean(dim='time')
        tmp = tmp.mean(dim=[latdim,londim],skipna=True)
        n34 = tmp.rolling(time=5,center=True).mean(dim='time')
        n34['tas'][:,0] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2])/3
        n34['tas'][:,1] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3])/4
        n34['tas'][:,2] = (tmp['tas'][:,0] + tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4])/5
        n34['tas'][:,3] = (tmp['tas'][:,1] + tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4] + tmp['tas'][:,5])/5
        n34['tas'][:,4] = (tmp['tas'][:,2] + tmp['tas'][:,3] + tmp['tas'][:,4] + tmp['tas'][:,5] + tmp['tas'][:,6])/5

        n34['tas'][:,-1] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3])/3
        n34['tas'][:,-2] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4])/4
        n34['tas'][:,-3] = (tmp['tas'][:,-1] + tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5])/5
        n34['tas'][:,-4] = (tmp['tas'][:,-2] + tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5] + tmp['tas'][:,-6])/5
        n34['tas'][:,-5] = (tmp['tas'][:,-3] + tmp['tas'][:,-4] + tmp['tas'][:,-5] + tmp['tas'][:,-6] + tmp['tas'][:,-7])/5
        print(n34)

        enso_ind = xr.Dataset(
        data_vars = dict(
                    eli=(['ens','time'], eli_mon),
                    n34=(['ens','time'], n34['tas'].data),
                    ),
                coords = dict(
                    time = nci['tas']['time'].data,
                    ens = variants,
                    ),
                attrs=dict(description='ELI and Nino 3.4 data from: '+model),
                    )
        print('|=====| Completed computation of ELI')

    nci['pr']['pr'] = nci['pr']['pr']*86400

    nco = nci['pr']['pr'].to_dataset()
    nco = nco.assign(tas=nci['tas']['tas'])
    nco = nco.assign(eli=enso_ind['eli'])
    nco = nco.assign(n34=enso_ind['n34'])

    nco.to_netcdf(diro+filo,mode='w',format='NETCDF4')
    print('Completed preprocessing of CMIP5 data for model: ' + model)

    # Explicit memory cleanup
    import gc
    del nco, nci, n34_sst, n34, troptas, troptasval, tmp, sstanom, enso_ind
    gc.collect()


def make_grid(lonmin,lonmax,latmin,latmax,res=1):
    dummygrid = xr.Dataset(
        data_vars = dict(
    ),
    coords = dict(
        lon = (['lon'], np.arange(lonmin,lonmax+res,res)),
        lat = (['lat'], np.arange(latmin,latmax+res,res)),
    ),
    )
    return dummygrid


def preproc_cmip_forWT(model, destgrid=None, latslic=slice(15,55), lonslic=slice(225,305),timslic=slice('1981-01-01','2016-12-31'), cmip='cmip5'):
    if destgrid is None:
        destgrid = make_grid(230,300,20,52)
    timstrt = timslic.start
    timend = timslic.stop

    if cmip == 'cmip5':
        variants = ['r1i1p1','r1i1p1','r6i1p1','r1i1p1','r1i1p1','r1i1p1']
        versions = ['v4','v20120410','latest','v20120710','v20120701','v20110901']
        models = ['ACCESS1-3','CanESM2','CCSM4','MIROC5','MRI-CGCM3','NorESM1-M']
        insts = ['CSIRO-BOM','CCCma','NCAR','MIROC','MRI','NCC']
    variables = ['hus','pr','psl','ta','ua','va','tas','tasmax','tasmin','zg']
    diro = '/glade/campaign/ral/hap/nlybarger/cmip5_preproc_for_WT/'
    try:
        imod = models.index(model)
    except ValueError:
        raise ValueError(f"Model '{model}' not found in models list.")
    
    variant = variants[imod]
    version = versions[imod]
    inst = insts[imod]
    diri = f'/glade/collections/cmip/{cmip}/output1/'

    for ivar, var in enumerate(variables):
        # Some models have version mismatches between historical and rcp45
        # So need to hardcode exceptions for these cases
        # Also had to download some data from ESGF myself
        filis1 = sorted(glob.glob(f'{diri}{inst}/{model}/historical/day/atmos/day/{variant}/{version}/{var}/*'))
        if model == 'NorESM1-M':
            filis2 = sorted(glob.glob(f'{diri}{inst}/{model}/rcp45/day/atmos/day/{variant}/v20121031/{var}/*'))
        elif model == 'CanESM2':
            filis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/cmip5_raw/CanESM_r1i1p1_rcp45_day_raw/{var}*.nc'))
        elif model == 'ACCESS1-3':
            filis2 = sorted(glob.glob(f'{diri}{inst}/{model}/rcp45/day/atmos/day/{variant}/v20131108/{var}/*'))
        elif model == 'MIROC5':
            filis2 = sorted(glob.glob(f'/glade/campaign/ral/hap/nlybarger/cmip5_raw/MIROC5_r1i1p1_rcp85_day_raw/{var}_day*'))
        else:
            filis2 = sorted(glob.glob(f'{diri}{inst}/{model}/rcp45/day/atmos/day/{variant}/{version}/{var}/*'))
        
        # Some models had superfluous files floating around
        for fil in filis1:
            if ('_19790101-20051231.nc' in fil) or ('_19500102-19891231.nc' in fil):
                filis1.remove(fil)
            if (model == 'MIROC5') and ('_20100101-20121231.nc' in fil):
                filis1.remove(fil)
        for fil in filis2:
            if (model == 'MIROC5') and ('_20060101-20091231.nc' in fil):
                    filis2.remove(fil)
            

        print(f'Beginning preprocessing of {var} for {model}')
        if ivar == 0:
            ds1 = xr.open_mfdataset(filis1,engine='netcdf4').sel(time=timslic, drop=True).load()
            ds2 = xr.open_mfdataset(filis2,engine='netcdf4').sel(time=timslic, drop=True).load()
            nci = xr.concat([ds1,ds2],dim='time')[[var]]
        else:
            ds1 = xr.open_mfdataset(filis1,engine='netcdf4').sel(time=timslic, drop=True).load()
            ds2 = xr.open_mfdataset(filis2,engine='netcdf4').sel(time=timslic, drop=True).load()
            tmp = xr.concat([ds1,ds2],dim='time')[[var]]

            # If regridder isn't defined, this will raise a NameError.
            # Similarly, if 'xxx' is not defined, 'xxx is None' will raise NameError.
            # To check if a variable exists, use 'if "xxx" in locals()' or 'globals()'.
            if ((tmp['lat'][1] != nci['lat'][1]) or (tmp['lon'][1] != nci['lon'][1])) and 'regridder' not in locals():
                regridder = xesmf.Regridder(tmp, nci, method='bilinear')
                tmp = regridder(tmp)
            elif ((tmp['lat'][1] != nci['lat'][1]) or (tmp['lon'][1] != nci['lon'][1])) and 'regridder' in locals():
                tmp = regridder(tmp)

            if var in ['pr', 'psl', 'tas', 'tasmax', 'tasmin']:
                nci[var] = (['time','lat','lon'], tmp[var].data)
            else:
                nci[var] = (['time','plev','lat','lon'], tmp[var].data)
    nci = nci.sel(lat=latslic,lon=lonslic,drop=True)
    if 'plev' in nci.dims or 'plev' in nci.coords:
        if not np.all(np.diff(nci['plev'].values)[::-1] <= 0):
            nci = nci.reindex(plev=list(reversed(nci['plev'])))

    # Standardize the time axis across models
    if model in ['CanESM2', 'CCSM4', 'MIROC5', 'NorESM1-M']:
        time = xr.date_range(start=timstrt, end=timend, freq='D', calendar='noleap', use_cftime=True)
    else:
        time = xr.date_range(start=timstrt, end=timend, freq='D', calendar='standard', use_cftime=True)
    nci = nci.assign_coords(time=time)

    # Compute additional variables for models defined for all vertical levels
    # These are computed for the other models in interp_nans_cmip5()
    if model not in ['ACCESS1-3','CCSM4','MIROC5','MRI-CGCM3']:
        dp = abs(nci['plev'].diff(dim='plev'))
        nci['pw'] = ((nci['hus'] * dp).sum(dim='plev')/9.81)
        nci['qflux'] = np.sqrt(((nci['hus']*nci['ua'])**2 + (nci['hus']*nci['va'])**2))
        nci['uva'] = np.sqrt((nci['ua']**2 + nci['va']**2))
    nci_grid_wb = i2g.get_latlon_b_rect(nci,'lon','lat','lon','lat')
    destgrid_wb = i2g.get_latlon_b_rect(destgrid,'lon','lat','lon','lat')
    regr = xesmf.Regridder(nci_grid_wb,destgrid_wb,'bilinear')
    nco = regr(nci)

    # These models fill vertical levels below terrain with NaNs, 
    # so we interpolate the lower levels from higher levels
    if model in ['ACCESS1-3','CCSM4','MIROC5','MRI-CGCM3']:
        nco = interp_nans_cmip5(nco)
    
    # MIROC5 is missing ZG for rcp45, so we pad the historical period with rcp85 instead
    if model not in ['MIROC5']:
        filo = f'{model}.historical_rcp45.{timstrt[:4]}-{timend[:4]}.WTpreproc.nc'
    else:
        filo = f'{model}.historical_rcp85.{timstrt[:4]}-{timend[:4]}.WTpreproc.nc'
    nco.to_netcdf(f'{diro}{filo}',mode='w',format='NETCDF4')


def interp_nans_cmip5(nci):
    # diri = '/glade/scratch/nlybarger/data/climate_data/cmip5/historical/'
    # nci = xr.open_dataset(diri + mod + '.conus.historical.nc')
    nci = nci.reindex(plev=list(reversed(nci['plev'])))
    ntim = len(nci['time'])
    nlev = len(nci['plev'])
    nlat = len(nci['lat'])
    nlon = len(nci['lon'])
    int_ua = np.zeros((ntim,nlev,nlat,nlon))
    int_va = np.zeros((ntim,nlev,nlat,nlon))
    int_hus = np.zeros((ntim,nlev,nlat,nlon))
    int_ta = np.zeros((ntim,nlev,nlat,nlon))
    int_zg = np.zeros((ntim,nlev,nlat,nlon))
    for it in range(ntim):
        # if it%100 == 0:
        #     print('Progress: ' + str(round(it*100/ntim,2)))
        int_ua[it,:,:,:]  = nci['ua'][it,:,:,:].interpolate_na(dim='plev', method='linear', fill_value='extrapolate')
        int_va[it,:,:,:]  = nci['va'][it,:,:,:].interpolate_na(dim='plev', method='linear', fill_value='extrapolate')
        int_hus[it,:,:,:] = nci['hus'][it,:,:,:].interpolate_na(dim='plev', method='linear', fill_value='extrapolate')
        int_ta[it,:,:,:]  = nci['ta'][it,:,:,:].interpolate_na(dim='plev', method='linear', fill_value='extrapolate')
        int_zg[it,:,:,:]  = nci['zg'][it,:,:,:].interpolate_na(dim='plev', method='linear', fill_value='extrapolate')

    nci['qflux'] = (['time','plev','lat','lon'], np.sqrt((int_hus*int_ua)**2 + (int_hus*int_va)**2))
    nci['uva']   = (['time','plev','lat','lon'], np.sqrt(int_ua**2 + int_va**2))
    nci['va']    = (['time','plev','lat','lon'], int_va)
    nci['ua']    = (['time','plev','lat','lon'], int_ua)
    nci['ta']    = (['time','plev','lat','lon'], int_ta)
    nci['zg']    = (['time','plev','lat','lon'], int_zg)

    nci['hus']   = (['time','plev','lat','lon'], int_hus)
    dp = abs(nci['plev'].diff(dim='plev'))
    nci['pw']    = (['time','lat','lon'], ((nci['hus'] * dp).sum(dim='plev')/9.81).data)
    nci = nci.reindex(plev=list(reversed(nci['plev'])))
    
    return nci
    # nci.to_netcdf(diri+mod+'.conus.historical.plevint.nc',mode='w')


def make_era5_daily(yr, im):
    diri = '/glade/campaign/collections/gdex/data/d633000/'
    pldiri = f'{diri}e5.oper.an.pl/'
    sfcdiri = f'{diri}e5.oper.an.sfc/'
    # prdiri = f'{diri}e5.oper.fc.sfc.accumu/'
    prdiri = f'/glade/campaign/ral/hap/nlybarger/OBS/ERA5/daily/'

    plvars = ['Q','V','U','T','Z']
    numsig = {'Q':'.128_133_q.ll025sc.', 'V':'.128_132_v.ll025uv.',
              'U':'.128_131_u.ll025uv.', 'T':'.128_130_t.ll025sc.',
              'Z':'.128_129_z.ll025sc.', 'TCW':'.128_136_tcw.ll025sc.',
              'PSL':'.128_151_msl.ll025sc.', 'T2M':'.128_167_2t.ll025sc.'}
    plevs = [850, 500, 250]
    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    yrstr = str(yr)
    diro = '/glade/derecho/scratch/nlybarger/era5_daily/'
    print(f'Beginning to process year: {yrstr}')



    # for im in np.arange(1,13):

    monstr = f'{im:02d}'

    if im == 2 and (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)):
        tmpmon = 29
        prevtmpmon = 31
    elif im == 3 and (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)):
        tmpmon = 31
        prevtmpmon = 29
    else:
        tmpmon = month_length[im-1]
        prevtmpmon = month_length[im-2]

    filo = f'era5_conus_daily_{yrstr}{monstr}_1deg.nc'
    print(f'|==| Beginning to process month: {monstr}')
    for ivar, var in enumerate(plvars):
        filis = sorted(glob.glob(f'{pldiri}/{yrstr}{monstr}/e5.oper.an.pl{numsig[var]}{yrstr}{monstr}*.nc'))
        # lonslic = slice(-130,-60)
        lonslic = slice(225,305)
        # latslic = slice(20,52)
        latslic = slice(55,15)
        tmp = xr.open_mfdataset(filis, engine='netcdf4').sel(longitude=lonslic, latitude=latslic, level=plevs).load()
        if ivar == 0:
            nci = xr.Dataset(
                coords={k: v for k, v in tmp.coords.items()},
                data_vars={k: v for k, v in tmp.data_vars.items() if k != var})
        for plev in plevs:
            if var != 'Z':
                nci[f'{var}{plev}'] = tmp[var].sel(level=plev)
            else:
                nci[f'ZG{plev}'] = tmp[var].sel(level=plev)/9.80665
    for plev in plevs:
        nci[f'MFL{plev}'] = np.sqrt((nci[f'U{plev}']*nci[f'Q{plev}'])**2 + (nci[f'V{plev}']*nci[f'Q{plev}'])**2)
        nci[f'UV{plev}'] = np.sqrt(nci[f'U{plev}']**2 + nci[f'V{plev}']**2)
    pwfili = f'{sfcdiri}/{yrstr}{monstr}/e5.oper.an.sfc{numsig['TCW']}{yrstr}{monstr}0100_{yrstr}{monstr}{tmpmon}23.nc'
    nci['PW'] = xr.open_dataset(pwfili, engine='netcdf4').sel(longitude=lonslic, latitude=latslic)['TCW'].load()

    pslfili = f'{sfcdiri}/{yrstr}{monstr}/e5.oper.an.sfc{numsig['PSL']}{yrstr}{monstr}0100_{yrstr}{monstr}{tmpmon}23.nc'
    nci['PSL'] = xr.open_dataset(pslfili, engine='netcdf4').sel(longitude=lonslic, latitude=latslic)['MSL'].load()

    t2mfili = f'{sfcdiri}/{yrstr}{monstr}/e5.oper.an.sfc{numsig['T2M']}{yrstr}{monstr}0100_{yrstr}{monstr}{tmpmon}23.nc'
    nci['T2M'] = xr.open_dataset(t2mfili, engine='netcdf4').sel(longitude=lonslic, latitude=latslic)['VAR_2T'].load()-273.15

    prfili = f'{prdiri}era5_daily_tp_{yrstr}.nc'
    pr = xr.open_dataset(prfili, engine='netcdf4')['tp'].load()*1000.  # convert from m to mm/day
    pr = pr.rename({'valid_time':'time', 'latitude':'lat', 'longitude':'lon'})
    pr = pr.reindex(lat=list(reversed(pr.lat)))
    pr['lon'] = pr['lon']+360.

    ### ==========================================================================================
    ### This computes daily total precipitation from accumulated 1-hourly data forecast files
    ### Because gdex doesn't store total precipitation analysis files
    ### ==========================================================================================

    # prevmonstr = '12' if monstr == '01' else f'{int(monstr)-1:02d}'
    # prevyrstr = f'{int(yrstr)-1:04d}' if monstr == '01' else yrstr
    # nextmonstr = f'{int(monstr)+1:02d}' if monstr != '12' else '01'
    # nextyrstr = yrstr if monstr != '12' else f'{int(yrstr)+1:04d}'

    # # have to load from previous month to get full monthly coverage
    # cp_filis1 = sorted(glob.glob((f'{prdiri}{prevyrstr}{prevmonstr}/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.*.nc')))
    # cp_filis2 = sorted(glob.glob((f'{prdiri}{yrstr}{monstr}/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.*.nc')))
    # cp_filis = (cp_filis1 + cp_filis2)[1:]
    # lsp_filis1 = sorted(glob.glob((f'{prdiri}{prevyrstr}{prevmonstr}/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.*.nc')))
    # lsp_filis2 = sorted(glob.glob((f'{prdiri}{yrstr}{monstr}/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.*.nc')))
    # lsp_filis = (lsp_filis1 + lsp_filis2)[1:]
    # cp = xr.open_mfdataset(cp_filis, engine='netcdf4').sel(longitude=lonslic, latitude=latslic).load()
    # lsp = xr.open_mfdataset(lsp_filis, engine='netcdf4').sel(longitude=lonslic, latitude=latslic).load()
    # tp = cp['CP']+lsp['LSP']
    # tp = tp.sel(forecast_initial_time=slice(f'{prevyrstr}-{prevmonstr}-{prevtmpmon}T18:00:00', f'{yrstr}-{monstr}-{tmpmon}T18:00:00'))

    # # this stacks the forecast initial time and forecast hour dimensions into a single time dimension
    # # that is of the form [[initial time 1 + hour 1], [initial time 1 + hour 2], ... [initial time 2 + hour 1], ...]
    # # this puts everything into the correct chronological order
    # tp_stacked = tp.stack(time=('forecast_initial_time', 'forecast_hour')).transpose('time', 'latitude', 'longitude')
    # tp_stacked = tp_stacked.drop_vars(['forecast_initial_time', 'forecast_hour'])

    # # now we assign a continuous hourly time coordinate to this stacked array, resample to daily totals, and select the month of interest
    # tp_stacked = tp_stacked.assign_coords(time=pd.date_range(start=f'{prevyrstr}-{prevmonstr}-{prevtmpmon}-19:00:00',
    #                                                            end=f'{nextyrstr}-{nextmonstr}-01-06:00:00', freq='h'))
    # tp_stacked = tp_stacked.resample(time='D').sum()

    if monstr == '02':
        timslic = slice(f'{yrstr}-{monstr}-01', f'{yrstr}-{monstr}-28')
    else:
        timslic = slice(f'{yrstr}-{monstr}-01', f'{yrstr}-{monstr}-{tmpmon}')
    ### ==========================================================================================

    nci = nci.rename({'latitude':'lat', 'longitude':'lon'})
    if nci['lat'][0] > nci['lat'][-1]:
        nci = nci.reindex(lat=list(reversed(nci.lat)))
    tasmax = nci['T2M'].resample(time='1D').max()
    tasmin = nci['T2M'].resample(time='1D').min()
    ncid = nci.resample(time='1D').mean()
    ncid = ncid.sel(time=timslic)
    no_leap = xr.date_range(start=timslic.start, end=timslic.stop, freq='D', calendar='noleap', use_cftime=True)
    ncid = ncid.assign_coords(time=no_leap)
    ncid['T2Mmax'] = tasmax
    ncid['T2Mmin'] = tasmin
    # ncid['PR'] = (['time', 'lat', 'lon'], tp_stacked.data*1000.)  # convert from m to mm/day
    ncid['PR'] = (['time', 'lat', 'lon'], pr.sel(time=timslic).data)


    destgrid = make_grid(230,300,20,52)
    nci_grid_wb = i2g.get_latlon_b_rect(ncid,'lon','lat','lon','lat')
    destgrid_wb = i2g.get_latlon_b_rect(destgrid,'lon','lat','lon','lat')
    regr = xesmf.Regridder(nci_grid_wb,destgrid_wb,'conservative_normed')
    ncid_1d = regr(ncid)
    ncid_1d.to_netcdf(diro+filo,mode='w')

    del nci
    del ncid
    del ncid_1d


def prep_cesm2le_forWT(ensind=0, expind=0):
    varlist = ['TREFHT','TREFHTMN','TREFHTMX',
            'PRECT','PS','PSL','TMQ',
            'T850','T700','T500','T200',
            'U850','U700','U500','U200',
            'V850','V700','V500','V200',
            'Q850','Q700','Q500','Q200',
            'Z850','Z700','Z500','Z200']
    TimeStart = '1981-01-01'
    TimeEnd = '2016-12-31'
    timslic = slice(TimeStart,TimeEnd)
    enslist = frdc5.CESM2_LE_ensemble_list()
    diri = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
    testfili = sorted(glob.glob(f'{diri}Z200/*BHIST*'))[0]
    testpr = xr.open_dataset(testfili,engine='netcdf4').isel(lat=0,lon=0,time=0)
    dvars = []
    for var in list(testpr.variables):
        if (var in varlist) or (var in ['lat','lon','time','date']):
            continue
        else:
            dvars.append(var)
    del testpr

    experiments = ['hist', 'ssp370']

    exp = experiments[expind]
    ens = enslist[ensind]
    lonslic = slice(225,305)
    latslic = slice(15,55)

    nci = {}
    diro = '/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/'

    for var in varlist:
        if exp == 'hist':
            filis = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/{var}/*BHIST*{ens}*'))[-9:]
            ds1 = xr.open_mfdataset(filis, engine='netcdf4', drop_variables=dvars).sel(time=slice(TimeStart, '2015-12-31'), lat=latslic, lon=lonslic).load()
            filis2 = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/{var}/*BSSP370*{ens}*'))[0]
            ds2 = xr.open_mfdataset(filis2, engine='netcdf4', drop_variables=dvars).sel(time=slice('2016-01-01', TimeEnd), lat=latslic, lon=lonslic).load()
            # filis.append(filis2)
            # ds = xr.open_mfdataset(filis, engine='netcdf4', drop_variables=dvars).sel(time=timslic, lat=latslic, lon=lonslic).load()
            ds = xr.concat([ds1,ds2],dim='time').sel(time=timslic)
            del ds1
            del ds2
        else:
            filis = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/{var}/*BSSP370*{ens}*'))
            ds = xr.open_mfdataset(filis, engine='netcdf4', drop_variables=dvars).sel(lat=latslic, lon=lonslic).load()
        if var == 'TREFHT':
            nci = ds.copy()
        else:
            nci[var] = ds[var]
        del ds
        del filis
    # nci.to_netcdf(f'{diro}CESM2-LE.{ens}.CONUS.{exp}.xprvars.nc')

    nci = nci.rename({'PRECT':'pr','TREFHT':'tas','TREFHTMN':'tasmin','TREFHTMX':'tasmax'})
    nci['pr'] = nci['pr']*86400*1000
    nci['pr'].attrs['units'] = 'mm'
    nci['pr'].attrs['long_name'] = 'Daily Total Precipitation'

    nci['tas'] = nci['tas']-273.15
    nci['tas'].attrs['units'] = 'degC'
    nci['tas'].attrs['long_name'] = 'Daily average 2m air temperature'

    nci['tasmin'] = nci['tasmin']-273.15
    nci['tasmin'].attrs['units'] = 'degC'
    nci['tasmin'].attrs['long_name'] = 'Daily minimum 2m air temperature'

    nci['tasmax'] = nci['tasmax']-273.15
    nci['tasmax'].attrs['units'] = 'degC'
    nci['tasmax'].attrs['long_name'] = 'Daily maximum 2m air temperature'
    # nci = nci.sel(time=slice('1930-01-01', '2014-12-31'))

    destgrid = make_grid(230,300,20,52)
    nci_grid_wb = i2g.get_latlon_b_rect(nci,'lon','lat','lon','lat')
    destgrid_wb = i2g.get_latlon_b_rect(destgrid,'lon','lat','lon','lat')
    regr = xesmf.Regridder(nci_grid_wb,destgrid_wb,'biliinear')
    nci_1d = regr(nci)
    nci_1d.to_netcdf(f'{diro}CESM2-LE_WTvars_CONUS_{ens}_{TimeStart[:4]}-{TimeEnd[:4]}.hist_1deg.nc')

    n34_latslic = slice(-5,5)
    n34_lonslic = slice(190,240)
    filis = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/month_1/TREFHT/*BHIST*{ens}*'))[-4:]
    filis2 = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/month_1/TREFHT/*BSSP370*{ens}*'))[0]
    filis.append(filis2)
    ncin34 = xr.open_mfdataset(filis,engine='netcdf4').sel(lat=n34_latslic,lon=n34_lonslic).load()
    ncin34 = ncin34.sel(time=timslic)
    nci_clim = ncin34.mean(dim='time')
    nci_anoms = ncin34['TREFHT']-nci_clim['TREFHT']
    n34 = nci_anoms.mean(dim=['lat','lon'])
    n34_smooth = n34.rolling(time=5, min_periods=1,center=True).mean()
    n34 = n34_smooth.to_dataset(name='n34')

    n34.to_netcdf(f'{diro}CESM2-LE_nino34_{ens}_1981-2016.nc',mode='w',format='NETCDF4')


# def combine_cesm2le_forWT(ens, timslic, scenario='hist'):
#     diri = '/glade/campaign/ral/hap/nlybarger/CESM2-LE_dseval/'
#     # if scenario == 'hist':
#     #     # diri = diri + 'hist/'
#     #    timstrt = '1981-01-01'
#     #    timend = '2016-12-31'
#     # elif scenario == 'ssp370':
#     #     # diri = diri + 'ssp370/'
#     #     timstrt = '2015-01-01'
#     #     timend = '2099-12-31'
#     # timslic = slice(timstrt, timend)
#     timstrt = str(timslic.start)
#     timend = str(timslic.stop)

#     # enslist = frdc5.CESM2_LE_ensemble_list()

#     # for iens,ens in enumerate(enslist):
#     fili = f'{diri}CESM2-LE.{ens}.CONUS.{scenario}.xprvars.nc'
#     # if ens == '1001.001':
#     nci = xr.open_dataset(fili,engine='netcdf4').load()
#         # nci = nci.expand_dims(dim={'ens':[ens]})
#     # else:
#     #     nci = xr.concat([nci, xr.open_dataset(fili,engine='netcdf4').load().expand_dims(dim={'ens':[ens]})],dim='ens')
#     # del fili

#     nci = nci.rename({'PRECT':'pr','TREFHT':'tas','TREFHTMN':'tasmin','TREFHTMX':'tasmax'})
#     nci['pr'] = nci['pr']*86400*1000
#     nci['pr'].attrs['units'] = 'mm'
#     nci['pr'].attrs['long_name'] = 'Daily Total Precipitation'

#     nci['tas'] = nci['tas']-273.15
#     nci['tas'].attrs['units'] = 'degC'
#     nci['tas'].attrs['long_name'] = 'Daily average 2m air temperature'

#     nci['tasmin'] = nci['tasmin']-273.15
#     nci['tasmin'].attrs['units'] = 'degC'
#     nci['tasmin'].attrs['long_name'] = 'Daily minimum 2m air temperature'

#     nci['tasmax'] = nci['tasmax']-273.15
#     nci['tasmax'].attrs['units'] = 'degC'
#     nci['tasmax'].attrs['long_name'] = 'Daily maximum 2m air temperature'
#     # nci = nci.sel(time=slice('1930-01-01', '2014-12-31'))

#     destgrid = make_grid(230,300,20,52)
#     nci_grid_wb = i2g.get_latlon_b_rect(nci,'lon','lat','lon','lat')
#     destgrid_wb = i2g.get_latlon_b_rect(destgrid,'lon','lat','lon','lat')
#     regr = xesmf.Regridder(nci_grid_wb,destgrid_wb,'conservative')
#     nci_1d = regr(nci)
#     nci_1d.to_netcdf(f'{diri}CESM2-LE_tas_pr_CONUS_{ens}_{timstrt[:4]}-{timend[:4]}.{scenario}_1deg.nc')

#     n34_latslic = slice(-5,5)
#     n34_lonslic = slice(190,240)
#     filis = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/month_1/TREFHT/*BHIST*{ens}*'))[-4:]
#     filis2 = sorted(glob.glob(f'/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/month_1/TREFHT/*BSSP370*{ens}*'))[0]
#     filis.append(filis2)
#     ncin34 = xr.open_mfdataset(filis,engine='netcdf4').sel(lat=n34_latslic,lon=n34_lonslic).load()
#     ncin34 = ncin34.sel(time=slice('1981-01-01', '2016-12-31'))
#     nci_clim = ncin34.mean(dim='time')
#     nci_anoms = ncin34['TREFHT']-nci_clim['TREFHT']
#     n34 = nci_anoms.mean(dim=['lat','lon'])
#     n34_smooth = n34.rolling(time=5, min_periods=1,center=True).mean()
#     n34 = n34_smooth.to_dataset(name='n34')
#     n34.to_netcdf(f'{diri}CESM2-LE_nino34_{ens}_1981-2016.nc',mode='w',format='NETCDF4')


def download_era5_daily_tp(year):
    ## MUST RUN FROM CONDA ENV CDSAPI
    import cdsapi
    # for year in range(1997, 2017):
    # year = 1998
    yrstr = f'{year:04d}'
    print(f'Requesting ERA5 Daily Data for Year: {year}')

    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": yrstr,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_sum",
        "time_zone": "utc+00:00",
        "frequency": "3_hourly",
        "area": [55, -135, 15, -55]
    }
    target = f"/glade/campaign/ral/hap/nlybarger/OBS/ERA5/daily/era5_daily_tp_{year}.nc"
    client = cdsapi.Client()
    client.retrieve(dataset, request, target)