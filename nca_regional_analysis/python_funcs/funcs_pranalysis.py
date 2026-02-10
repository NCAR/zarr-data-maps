import matplotlib.path as mplPath
import shapefile as shp
import pickle
import glob
import numpy as np
import pandas as pd
import os
import shapefile
import xarray as xr
import datetime as dt
import funcs_general as fg
import funcs_read_ds_cmip5 as frdc5


def compute_freezing_levels(temp, pres, geopot_height):
    """
    Compute the freezing level height and wet bulb freezing level height given 3D profiles of temperature, pressure, and geopotential height.

    Parameters:
    temp (numpy.ndarray): 3D array of temperature profiles (K)
    pres (numpy.ndarray): 3D array of pressure profiles (Pa)
    geopot_height (numpy.ndarray): 3D array of geopotential height profiles (m)

    Returns:
    tuple: Two 2D arrays representing the freezing level height and wet bulb freezing level height (m)
    """

    # Convert temperature to Celsius and pressure to hPa
    temp_c = temp - 273.15
    pres_hpa = pres / 100.0

    # Initialize arrays for freezing level height and wet bulb freezing level height
    freezing_level_height = np.full(temp.shape[:2], np.nan)
    wet_bulb_freezing_level_height = np.full(temp.shape[:2], np.nan)

    # Loop over each vertical profile
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            # Get the temperature, pressure, and geopotential height profile for the current grid point
            temp_profile = temp_c[i, j, :]
            pres_profile = pres_hpa[i, j, :]
            height_profile = geopot_height[i, j, :]

            # Compute the wet bulb temperature profile
            wet_bulb_profile = wet_bulb_temperature(pres_profile * units.hPa, temp_profile * units.degC, temp_profile * units.degC)

            # Find the freezing level height (where temperature crosses 0°C)
            freezing_level_idx = np.where(temp_profile <= 0)[0]
            if len(freezing_level_idx) > 0:
                freezing_level_height[i, j] = height_profile[freezing_level_idx[0]]

            # Find the wet bulb freezing level height (where wet bulb temperature crosses 0°C)
            wet_bulb_freezing_level_idx = np.where(wet_bulb_profile.magnitude <= 0)[0]
            if len(wet_bulb_freezing_level_idx) > 0:
                wet_bulb_freezing_level_height[i, j] = height_profile[wet_bulb_freezing_level_idx[0]]
            else:
                # If the wet bulb freezing level is not found, set it to the freezing level
                wet_bulb_freezing_level_height[i, j] = freezing_level_height[i, j]
    return freezing_level_height, wet_bulb_freezing_level_height


def MakeShapefile_watershed(LonW,LatW,sShapefiles,domain='TaylorPark'):
    rgrGridCells=[(LonW.ravel()[ii],LatW.ravel()[ii]) for ii in range(len(LonW.ravel()))]
    watershed_WRF=np.zeros((LonW.shape[0]*LonW.shape[1]))

    sf = shp.Reader(sShapefiles)
    if domain == 'LahontanValley':
        df = read_shapefile(sf,tform=True)
    else:
        df = read_shapefile(sf)
    
    for sf in range(df.shape[0]):
        ctr = df['coords'][sf]
        if len(ctr) > 10000:
            ctr=np.array(ctr)[::100,:] # coarsen the shapefile accuracy
        else:
            ctr=np.array(ctr)
        grPRregion=mplPath.Path(ctr)
        TMP=np.array(grPRregion.contains_points(rgrGridCells))
        watershed_WRF[TMP == 1]=1
    watershed_WRF=np.reshape(watershed_WRF, (LatW.shape[0], LatW.shape[1]))
    return watershed_WRF

def MakeShapefile_HUC2(Regions,LonW,LatW,sShapefiles):
    rgrGridCells=[(LonW.ravel()[ii],LatW.ravel()[ii]) for ii in range(len(LonW.ravel()))]
    HUC2_WRF=np.zeros((LonW.shape[0]*LonW.shape[1]))
    for bs in range(len(Regions)):
        Basins = [Regions[bs]]
        for ba in range(len(Basins)):
            # print('        process '+Basins[ba])
            sf = shp.Reader(sShapefiles+Basins[ba])
            df = read_shapefile(sf)
            for sf in range(df.shape[0]):
                ctr = df['coords'][sf]
                if len(ctr) > 10000:
                    ctr=np.array(ctr)[::100,:] # carsen the shapefile accuracy
                else:
                    ctr=np.array(ctr)
                grPRregion=mplPath.Path(ctr)
                TMP=np.array(grPRregion.contains_points(rgrGridCells))
                HUC2_WRF[TMP == 1]=bs+1
    HUC2_WRF=np.reshape(HUC2_WRF, (LatW.shape[0], LatW.shape[1]))
    return HUC2_WRF

def read_shapefile(sf,tform=False):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    if tform:
        import pyproj
        sShapefiles = '/glade/u/home/nlybarger/shapefiles/BOR_Contributing_Area.shp'
        prj_file_path = sShapefiles.replace('.shp', '.prj')
        sf = shapefile.Reader(sShapefiles)
        df = read_shapefile(sf)
        with open(prj_file_path, 'r') as prj_file:
            projection_info = prj_file.read()
        current_projection = pyproj.CRS(projection_info)
        wgs84_projection = pyproj.CRS('EPSG:4326')  # WGS84
        transformer = pyproj.Transformer.from_crs(current_projection, wgs84_projection, always_xy=True)
        
        shpsnew = [s.points for s in sf.shapes()]
        for s in range(len(shps[0])):
            shpsnew[0][s] = transformer.transform(shps[0][s][0],shps[0][s][1])
        shps = shpsnew
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def read_case(domain,experiment,accum_period,ensemble_member,dvars):
    diri1 = '/glade/derecho/scratch/nlybarger/data/CESM2-LE/cesm2_wrfi/'
    diri2 = '/glade/campaign/ral/nral0041/'
    first = True
    second = True
    filis = sorted(glob.glob(diri1 + domain + '/' + experiment + '/' + accum_period 
                         + '/' + ensemble_member + '*/WRF/runwrf/run/wrf2d_d01_*'))
    if not filis:
        filis = sorted(glob.glob(diri2 + domain + '/' + experiment + '/' + accum_period 
                             + '/' + ensemble_member + '*/WRF/runwrf/run/wrf2d_d01_*'))
    if not filis:
        filis = sorted(glob.glob(diri1 + domain + '/' + experiment + '/' + accum_period 
                             + '/' + ensemble_member + '*/WRF/wrf2d_d01_*'))
    if not filis:
        return None
    # Leaves off the first and last days of the WRF simulations
    for fil in filis[23:-25]:
        if first:
            frsttim = xr.open_dataset(fil,engine='netcdf4',drop_variables=dvars)
            datout = xr.open_dataset(fil,engine='netcdf4',drop_variables=dvars)
            first = False
        else:
            tmp = xr.open_dataset(fil,engine='netcdf4',drop_variables=dvars)
            datout = xr.concat([datout,tmp],dim='Time')
    varlist = ['RAINNC','SNOWNC','GRAUPELNC']
    datout[varlist] = datout[varlist]-frsttim[varlist].isel(Time=0)
    return datout


def give_xy(ycoord,xcoord,pltdom):
    lat = pltdom['XLAT'][0,ycoord,xcoord].data.item()
    lon = pltdom['XLONG'][0,ycoord,xcoord].data.item()
    return (lat,lon)


def prec_mets(domain,experiment,accumulation_period,dummyvars,HUC2mask,wsmask,lat,lon):

    outdom = f'{domain}.radius'

    diro = '/glade/work/nlybarger/DSO_pr_analysis/new/'
    testfilo = f'{outdom}.{experiment}.{accumulation_period}.prtot.pkl'
    if os.path.exists(diro+testfilo):
        print('pr analysis already performed for:')
        print(f'===| {domain}, {experiment}, {accumulation_period}')
        print('Continuing')
        return
    
    print('Beginning to run pr analysis for:')
    print(f'===| {domain}, {experiment}, {accumulation_period}')

    enslen = {}
    enslen['1.accum'] = 600
    enslen['3.accum'] = 400
    enslen['7.accum'] = 250
    enslen['15.accum'] = 100
    filslen = {}
    filslen['1.accum'] = (3*24)+1
    filslen['3.accum'] = (5*24)+1
    filslen['7.accum'] = (9*24)+1
    filslen['15.accum'] = (17*24)+1
    enss = []
    for i in range(1,enslen[accumulation_period]+1):
        enss.append(f'{i:03}')

    remlist = []
    for iens in range(len(enss)):
        fils = sorted(glob.glob(f'/glade/derecho/scratch/nlybarger/data/CESM2-LE/cesm2_wrfi/{domain}/{experiment}/{accumulation_period}/{enss[iens]}*/WRF/runwrf/run/wrf2d*'))
        if not fils:
            fils = sorted(glob.glob(f'/glade/campaign/ral/nral0041/{domain}/{experiment}/{accumulation_period}/{enss[iens]}*/WRF/runwrf/run/wrf2d*'))
        if not fils:
            fils = sorted(glob.glob(f'/glade/derecho/scratch/nlybarger/data/CESM2-LE/cesm2_wrfi/{domain}/{experiment}/{accumulation_period}/{enss[iens]}*/WRF/wrf2d*'))
        if not fils:
            remlist.append(enss[iens])
        elif (len(fils) < filslen[accumulation_period]):
            remlist.append(enss[iens])
    for ens in remlist:
        enss.remove(ens)

    
    datin = {}
    for ens in enss:
        print('===|===| Reading ensemble member: ' + ens)
        datin[ens] = read_case(domain, experiment, accumulation_period, ens, dummyvars)
    
    if domain == 'TaylorPark':
        huc2val = 14
    elif domain == 'LahontanValley':
        huc2val = 16
    huc2mask = xr.where(HUC2mask==huc2val,1,0)
    
    enskeys = list(datin.keys())
    ntim = len(datin[enskeys[0]]['Time'])
    nens = len(enss)
    nlat = lat.shape[0]
    nlon = lat.shape[1]
    scales = ['domain','basin','watershed']

    prtot = {}
    raintot = {}
    snowtot = {}
    grauptot = {}
    pr_hrly_max = {}
    pr_space_sd = {}
    pr_composite = {}
    rain_composite = {}
    snow_composite = {}
    for dom in scales:
        prtot[dom] = {}
        raintot[dom] = {}
        snowtot[dom] = {}
        grauptot[dom] = {}
        pr_hrly_max[dom] = {}
        pr_space_sd[dom] = {}
        if dom == 'basin':
            mask = huc2mask.copy()
        elif (domain == 'LahontanValley') and (dom == 'watershed'):
            mask = wsmask.copy()
        elif dom == 'domain':
            mask = 1
        for iens in range(nens):
            if (domain == 'TaylorPark') and (dom == 'watershed'):
                
                test = datin[ens]['RAINNC'][0,:,:]*huc2mask
                max_idx = np.unravel_index(np.nanargmax(test), test.shape)
                mask = fg.create_radius_mask(test.shape, max_idx, 3.2)


            ens = enss[iens]
            tmprain = datin[ens]['RAINNC'].copy()
            tmpsnow = datin[ens]['SNOWNC'].copy()
            tmpgrap = datin[ens]['GRAUPELNC'].copy()

            raintot[dom][ens] = np.squeeze(tmprain.sel(Time=[ntim-1])*mask)
            snowtot[dom][ens] = np.squeeze(tmpsnow.sel(Time=[ntim-1])*mask)
            grauptot[dom][ens] = np.squeeze(tmpgrap.sel(Time=[ntim-1])*mask)
            prtot[dom][ens] = raintot[dom][ens] + snowtot[dom][ens] + grauptot[dom][ens]
        
        # tmp2 = np.zeros((ntim,nlat,nlon))
        # tmp3 = np.zeros((ntim,nlat,nlon))
        # tmp4 = np.zeros((ntim,nlat,nlon))
        # for it in range(ntim):
        #     if it == 0:
        #         tmp2[it,:,:] = (tmp.sel(Time=[it]))
        #     else:
        #         tmp2[it,:,:] = (tmp.sel(Time=[it]) - tmp.sel(Time=[it-1]))
        #     tmp3[it,:,:] = tmp2[it,:,:]*huc2mask
        #     tmp4[it,:,:] = tmp2[it,:,:]*wsmask
        # pr_hrly_max['basin'][ens] = np.max(tmp3,axis=0)
        # pr_hrly_max['watershed'][ens] = np.max(tmp4,axis=0)
        # pr_hrly_max['domain'][ens] = np.max(tmp2,axis=0)
        #     if dom == 'domain':
        #         pr_space_sd[dom][ens] = np.nanstd(prtot[dom][ens])
        #     else:
        #         pr_space_sd[dom][ens] = np.nanstd(xr.where(prtot[dom][ens]==0,np.nan,prtot[dom][ens]))
            
        #     if iens == 0:
        #         rain_composite[dom] = raintot[dom][ens].copy()
        #         snow_composite[dom] = snowtot[dom][ens].copy()
        #         pr_composite[dom] = prtot[dom][ens].copy()
        #     else:
        #         rain_composite[dom] += raintot[dom][ens].copy()
        #         snow_composite[dom] += snowtot[dom][ens].copy()
        #         pr_composite[dom] += prtot[dom][ens].copy()

        # rain_composite[dom] = rain_composite[dom]/nens
        # snow_composite[dom] = snow_composite[dom]/nens
        # pr_composite[dom] = pr_composite[dom]/nens

    #==============================================================================

    filo = f'{outdom}.{experiment}.{accumulation_period}.raintot.pkl'
    with open(diro+filo, 'wb') as file:
        pickle.dump(raintot, file)

    filo = f'{outdom}.{experiment}.{accumulation_period}.snowtot.pkl'
    with open(diro+filo, 'wb') as file:
        pickle.dump(snowtot, file)

    filo = f'{outdom}.{experiment}.{accumulation_period}.prtot.pkl'
    with open(diro+filo, 'wb') as file:
        pickle.dump(prtot, file)

    #==============================================================================

    # filo = domain+'.'+experiment+'.'+accumulation_period+'.pr_hrly_max.pkl'
    # with open(diro+filo, 'wb') as file:
    #     pickle.dump(pr_hrly_max, file)
    
    # filo = domain+'.'+experiment+'.'+accumulation_period+'.pr_space_sd.pkl'
    # with open(diro+filo, 'wb') as file:
    #     pickle.dump(pr_space_sd, file)

    #==============================================================================

    # filo = domain+'.'+experiment+'.'+accumulation_period+'.rain_composite.pkl'
    # with open(diro+filo, 'wb') as file:
    #     pickle.dump(rain_composite, file)

    # filo = domain+'.'+experiment+'.'+accumulation_period+'.snow_composite.pkl'
    # with open(diro+filo, 'wb') as file:
    #     pickle.dump(snow_composite, file)

    # filo = domain+'.'+experiment+'.'+accumulation_period+'.pr_composite.pkl'
    # with open(diro+filo, 'wb') as file:
    #     pickle.dump(pr_composite, file)

    # ncdiro = '/glade/derecho/scratch/nlybarger/data/CESM2-LE/'
    # filo = domain+'.'+experiment+'.'+accumulation_period+'.wrf2d.pkl'
    # with open(ncdiro+filo, 'wb') as file:
    #     pickle.dump(datin, file)
    
    
    return #prtot,pr_hrly_max,pr_space_sd,pr_composite


def find_pr_extremes(latslic,latind,lonslic,lonind,prDataset,domainName,exp,nDayAccum=1,nxt=10,MinDistDD=7,topPercentile=0.10,monlist=np.arange(1,13),prstr='PRECT',latstr='lat',lonstr='lon',ADJUST_NXT=True):
    # latslic, lonslic : slice variables over which extremes are desired
    # prDataset : xarray Dataset which includes precipitation
    # nDayAccum : accumulation period in days
    # nxt : number of extreme events desired
    # MinDistDD : minimum distance between extremes in days
    # topPercentile : parameter used to determine nxt if minimum distance needs to be adjusted and ADJUST_NXT=True
    #                 otherwise nxt remains unchanged if minimum distance needs to be adjusted
    # prstr, latstr, lonstr : strings associated with the respective variables in the prDataset
    # ADJUST_NXT : boolean to determine whether to adjust nxt if minimum distance needs to be adjusted
    #
    # This function finds nxt extreme events within prDataset as the maximum lonslic,latslic averaged precipitation
    # over the number of days defined by nDayAccum.  It returns an array containing the time indices of the extremes.
    #
    # Requires numpy and xarray
    #
    # This function was adapted from Andreas Prein's extreme weather type function by
    # Nicholas D. Lybarger; August 22, 2023

    # latslic=tplat
    # lonslic=tplon
    # prDataset=pr
    # monlist=monlist
    # nDayAccum=nDayAccum
    # MinDistDD=MinDistDD
    # nxt=nxt
    # topPercentile=topPercentile
    # prstr='PRECT'
    # latstr='lat'
    # lonstr='lon'
    # ADJUST_NXT=True
    # allens = ['1011.001','1031.002','1051.003','1071.004','1091.005','1111.006','1131.007','1151.008','1171.009','1191.010']
    allens = frdc5.CESM2_LE_ensemble_list()
    if nDayAccum%2 != 1:
        print('Accumulation period must be an odd number of days')
        # return

    if MinDistDD < (nDayAccum/2):
        print('Minimum distance between extremes is less than half of the accumulation period.')
        print('This may cause overlap of days within respective extreme\'s accumulation window.')
        print('Accumulation period: ' + str(nDayAccum) + ', Minimum distance between extremes: ' + str(MinDistDD))
        print('Increasing minimum distance to be greater than half the accumulation period.')
        while MinDistDD <= (nDayAccum/2):
            MinDistDD += 1
        print('New minimum distance: ' + str(MinDistDD))
        nday = len(prDataset['PRECT']['time'])
        if ADJUST_NXT:
            print('Adjusting number of extremes in light of this new minimum distance.')
            print('Previous number of extremes: ' + str(nxt))
            nxt = int((nday*(len(monlist)/12)/(nDayAccum+(MinDistDD*2)))*topPercentile)*10
            if nxt%2 == 0:
                nxt = nxt-1
            print('New number of extremes: ' + str(nxt))
    TP = (domainName == 'TaylorPark')
    tppr = prDataset[prstr].copy().isel(lat=latslic,lon=lonslic).sel(time=prDataset['time.month'].isin(monlist))
    ntim = len(tppr['time'])
    nens = len(tppr['ens'])
    rgrPRrecords=tppr.mean(dim=[latstr,lonstr]).rolling(time=nDayAccum,center=True,min_periods=1).sum()
    SortedDates=np.argsort(rgrPRrecords.data,axis=None)[:][::-1]
    rgiExtremePR=np.full((nxt),np.nan)
    ii=0
    jj=0
    rgiExtremePR[0]=SortedDates[0]
    while ii < nxt:
        if np.nanmin(np.abs(rgiExtremePR - SortedDates[jj])) < MinDistDD:
            jj+=1
        elif (SortedDates[jj] < (nDayAccum/2)):
            jj+=1
        elif (SortedDates[jj]-len(SortedDates) > -(nDayAccum/2)):
            jj+=1
        else:
            rgiExtremePR[ii]=SortedDates[jj]
            jj=jj+1
            ii=ii+1
    rgiExtremePR=rgiExtremePR.astype('int')
    xtensi,xtdati=np.unravel_index(rgiExtremePR,(nens,ntim))

    # ensout = np.empty(nxt,dtype='S8')
    # datend = np.full(nxt,-9999)
    # datstrt = np.full(nxt,-9999)
    enslist = ['0']*nxt
    datstrt = [-9999]*nxt
    datend = [-9999]*nxt
    baddies = []
    tmp_xtensi = np.array([-9999]*nxt)
    tmp_xtdati = np.array([-9999]*nxt)
    tmp_datstrt = np.array([-9999]*nxt)
    tmp_datend = np.array([-9999]*nxt)
    tmp_enslist = ['0']*nxt
    # ['PRECC', 'PRECL', 'PRECT', 'PS', 'Q500', 'TMQ', 'Z500', 'T500','RH500']
    for i in range(nxt):
        # plt.scatter(i,rgrPRrecords[a[0][i],a[1][i]])
        # print(prDataset[exp]['RH500'].isel(lat=latind,lon=lonind,ens=xtensi[i],time=xtdati[i]).data.item())
        try:
            print(xtensi[i])
            enslist[i]=allens[xtensi[i]]
            timmy = tppr['time'][xtdati[i]]
            datmidtmp = str(prDataset['date'].sel(ens=allens[xtensi[i]],time=timmy).item())
            datmid = datmidtmp[0:4]+'-'+datmidtmp[4:6]+'-'+datmidtmp[6:8]
            tdelts = dt.timedelta(days=(((nDayAccum-1)/2)+1))
            tdelte = dt.timedelta(days=(((nDayAccum-1)/2)+2))
            datstrt[i] = prDataset['date'].sel(time=(prDataset['time'].sel(time=datmid)-tdelts))[0].data.astype(int)
            strtstr = str(datstrt[i])
            tmpdatstrt = strtstr[1:5] + '-' + strtstr[5:7] + '-' + strtstr[7:9]
            datend[i] = prDataset['date'].sel(time=(prDataset['time'].sel(time=datmid)+tdelte))[0].data.astype(int)
            endstr = str(datend[i])
            tmpdatend = endstr[1:5] + '-' + endstr[5:7] + '-' + endstr[7:9]
            if (datend[i] == -9999) or (datstrt[i] == -9999):
                print(i)
                baddies.append(i)
        except KeyError:
            baddies.append(i)
            continue
        if TP:
            if nDayAccum == 1:
                thresh = 83.
                tval = prDataset['RH700']['RH700'].isel(lat=latind,lon=lonind,ens=xtensi[i]).sel(time=tppr['time'][xtdati[i]]).data.item()
                if tval > thresh:
                    tmp_xtensi[i] = xtensi[i]
                    tmp_xtdati[i] = xtdati[i]
                    tmp_datstrt[i] = datstrt[i]
                    tmp_datend[i] = datend[i]
                    tmp_enslist[i] = allens[xtensi[i]]
            else:
                if nDayAccum == 3:
                    thresh = 3144.
                elif nDayAccum == 7:
                    thresh = 3165.
                elif nDayAccum == 15:
                    thresh = 9999.
                tslic = slice(tmpdatstrt,tmpdatend)
                tval = prDataset['Z700']['Z700'].isel(lat=latind,lon=lonind,ens=xtensi[i]).sel(time=tslic).mean(dim='time').data.item()
                if tval < thresh:
                    tmp_xtensi[i] = xtensi[i]
                    tmp_xtdati[i] = xtdati[i]
                    tmp_datstrt[i] = datstrt[i]
                    tmp_datend[i] = datend[i]
                    tmp_enslist[i] = allens[xtensi[i]]

    baddies = sorted(baddies)
    print('Baddy length:')
    print(len(baddies))
    j=0
    for i in baddies:
        enslist.pop(i-j)
        datstrt.pop(i-j)
        datend.pop(i-j)
        j+=1
    if TP:
        filt_xtensi  = tmp_xtensi[tmp_xtensi != -9999]
        filt_xtdati  = tmp_xtdati[tmp_xtdati != -9999]
        filt_datstrt = tmp_datstrt[tmp_datstrt != -9999]
        filt_datend  = tmp_datend[tmp_datend != -9999]
        filt_enslist = []
        for i in range(len(tmp_enslist)):
            if tmp_enslist[i] != '0':
                filt_enslist.append(tmp_enslist[i])
        print('Post-threshold cull length:')
        print(len(filt_datstrt))
    else:
        filt_xtensi = None
        filt_xtdati = None
        filt_datstrt = None
        filt_datend = None

    with open('./CESM2-LE_'+exp+'_'+domainName+'_'+str(nDayAccum)+'.accum.txt', "w") as f:
        #Print header
        f.write("StartDate EndDate EnsembleName\n")

        #Print data from NumPy arrays
        if TP:
            print(len(filt_datstrt))
            print(len(filt_datend))
            print(len(filt_enslist))
            for i in range(len(filt_datstrt)):
                if i in np.arange(9):
                    f.write(f"00{i+1} {str(filt_datstrt[i]).strip('[]')} {str(filt_datend[i]).strip('[]')} {filt_enslist[i]}\n")
                elif i in np.arange(9,99):
                    f.write(f"0{i+1} {str(filt_datstrt[i]).strip('[]')} {str(filt_datend[i]).strip('[]')} {filt_enslist[i]}\n")
                else:
                    f.write(f"{i+1} {str(filt_datstrt[i]).strip('[]')} {str(filt_datend[i]).strip('[]')} {filt_enslist[i]}\n")
        else:
            for i in range(len(datstrt)):
                if i in np.arange(9):
                    f.write(f"00{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")
                elif i in np.arange(9,99):
                    f.write(f"0{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")
                else:
                    f.write(f"{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")

    

    return xtensi,xtdati,filt_xtensi,filt_xtdati,filt_datstrt,filt_datend


def give_xy(ycoord, xcoord, domain, latin, lonin):
    if domain == 'LahontanValley':
        latout = latin[ycoord,xcoord].data.item()
        lonout = lonin[ycoord,xcoord].data.item()
    elif domain == 'TaylorPark':
        latout = latin[ycoord,xcoord].data.item()
        lonout = lonin[ycoord,xcoord].data.item()
    return (latout,lonout)


# def make_basemap(domain, subdomain, dset, ll_lat=np.nan, ll_lon=np.nan, ur_lat=np.nan, ur_lon=np.nan):
#     if (subdomain == 'domain') and (dset == 'WRF'):
#         if domain == 'LahontanValley':
#             m = Basemap(width=1600000,height=1600000,
#                     rsphere=(6378137.00,6356752.3142),\
#                     resolution='i',area_thresh=1000.,projection='lcc',\
#                     lat_1=30.,lat_2=45,lat_0=39.2,lon_0=-120.)
#         elif domain == 'TaylorPark':
#             m = Basemap(width=1800000,height=1800000,
#                     rsphere=(6378137.00,6356752.3142),\
#                     resolution='i',area_thresh=1000.,projection='lcc',\
#                     lat_1=30.,lat_2=45,lat_0=39.5,lon_0=-108.)
#     else:
#         m = Basemap(resolution='i',projection='merc',llcrnrlat=ll_lat,llcrnrlon=ll_lon,
#                     urcrnrlat=ur_lat,urcrnrlon=ur_lon)
#     return m

def find_pr_extremes_NOTP(latslic,lonslic,prDataset,domainName,exp,nDayAccum=1,nxt=10,MinDistDD=7,topPercentile=0.10,monlist=np.arange(1,13),prstr='PRECT',latstr='lat',lonstr='lon',ADJUST_NXT=True):
    # latslic, lonslic : slice variables over which extremes are desired
    # prDataset : xarray Dataset which includes precipitation
    # nDayAccum : accumulation period in days
    # nxt : number of extreme events desired
    # MinDistDD : minimum distance between extremes in days
    # topPercentile : parameter used to determine nxt if minimum distance needs to be adjusted and ADJUST_NXT=True
    #                 otherwise nxt remains unchanged if minimum distance needs to be adjusted
    # prstr, latstr, lonstr : strings associated with the respective variables in the prDataset
    # ADJUST_NXT : boolean to determine whether to adjust nxt if minimum distance needs to be adjusted
    #
    # This function finds nxt extreme events within prDataset as the maximum lonslic,latslic averaged precipitation
    # over the number of days defined by nDayAccum.  It returns an array containing the time indices of the extremes.
    #
    # Requires numpy and xarray
    #
    # This function was adapted from Andreas Prein's extreme weather type function by
    # Nicholas D. Lybarger; August 22, 2023

    # latslic=tplat
    # lonslic=tplon
    # prDataset=pr
    # monlist=monlist
    # nDayAccum=nDayAccum
    # MinDistDD=MinDistDD
    # nxt=nxt
    # topPercentile=topPercentile
    # prstr='PRECT'
    # latstr='lat'
    # lonstr='lon'
    # ADJUST_NXT=True
    # allens = ['1011.001','1031.002','1051.003','1071.004','1091.005','1111.006','1131.007','1151.008','1171.009','1191.010']
    allens = frdc5.CESM2_LE_ensemble_list(DAILY_ALLVERT=True)
    if nDayAccum%2 != 1:
        print('Accumulation period must be an odd number of days')
        # return

    if MinDistDD < (nDayAccum/2):
        print('Minimum distance between extremes is less than half of the accumulation period.')
        print('This may cause overlap of days within respective extreme\'s accumulation window.')
        print('Accumulation period: ' + str(nDayAccum) + ', Minimum distance between extremes: ' + str(MinDistDD))
        print('Increasing minimum distance to be greater than half the accumulation period.')
        while MinDistDD <= (nDayAccum/2):
            MinDistDD += 1
        print('New minimum distance: ' + str(MinDistDD))
        nday = len(prDataset['PRECT']['time'])
        if ADJUST_NXT:
            print('Adjusting number of extremes in light of this new minimum distance.')
            print('Previous number of extremes: ' + str(nxt))
            nxt = int((nday*(len(monlist)/12)/(nDayAccum+(MinDistDD*2)))*topPercentile)*10
            if nxt%2 == 0:
                nxt = nxt-1
            print('New number of extremes: ' + str(nxt))
    tppr = prDataset[prstr].copy().isel(lat=latslic,lon=lonslic).sel(time=prDataset['time.month'].isin(monlist))
    ntim = len(tppr['time'])
    nens = len(tppr['ens'])
    rgrPRrecords=tppr.mean(dim=[latstr,lonstr]).rolling(time=nDayAccum,center=True,min_periods=1).sum()
    SortedDates=np.argsort(rgrPRrecords.data,axis=None)[:][::-1]
    rgiExtremePR=np.full((nxt),np.nan)
    ii=0
    jj=0
    rgiExtremePR[0]=SortedDates[0]
    while ii < nxt:
        if np.nanmin(np.abs(rgiExtremePR - SortedDates[jj])) < MinDistDD:
            jj+=1
        elif (SortedDates[jj] < (nDayAccum/2)):
            jj+=1
        elif (SortedDates[jj]-len(SortedDates) > -(nDayAccum/2)):
            jj+=1
        else:
            rgiExtremePR[ii]=SortedDates[jj]
            jj=jj+1
            ii=ii+1
    rgiExtremePR=rgiExtremePR.astype('int')
    xtensi,xtdati=np.unravel_index(rgiExtremePR,(nens,ntim))

    # ensout = np.empty(nxt,dtype='S8')
    # datend = np.full(nxt,-9999)
    # datstrt = np.full(nxt,-9999)
    enslist = ['0']*nxt
    datstrt = [-9999]*nxt
    datend = [-9999]*nxt
    baddies = []
    # ['PRECC', 'PRECL', 'PRECT', 'PS', 'Q500', 'TMQ', 'Z500', 'T500','RH500']
    for i in range(nxt):
        # try:
        enslist[i]=allens[xtensi[i]]
        timmy = tppr['time'][xtdati[i]]
        datmidtmp = str(prDataset['date'].sel(ens=allens[xtensi[i]],time=timmy).item())
        datmid = datmidtmp[0:4]+'-'+datmidtmp[4:6]+'-'+datmidtmp[6:8]
        tdelts = dt.timedelta(days=(((nDayAccum-1)/2)+1))
        tdelte = dt.timedelta(days=(((nDayAccum-1)/2)+2))
        try:
            datstrt[i] = prDataset['date'].sel(time=(prDataset['time'].sel(time=datmid)-tdelts))[0].data.astype(int)
            datend[i] = prDataset['date'].sel(time=(prDataset['time'].sel(time=datmid)+tdelte))[0].data.astype(int)
        except KeyError:
            baddies.append(i)
            continue
        if (datend[i] == -9999) or (datstrt[i] == -9999):
            print(i)
            baddies.append(i)
        # except KeyError:
        #     baddies.append(i)
        #     continue

    baddies = sorted(baddies)
    print('Baddy length:')
    print(len(baddies))
    j=0
    for i in baddies:
        enslist.pop(i-j)
        datstrt.pop(i-j)
        datend.pop(i-j)
        xtdati = np.delete(xtdati, i-j)
        xtensi = np.delete(xtensi, i-j)
        j+=1

    with open('./CESM2-LE_'+exp+'_'+domainName+'_'+str(nDayAccum)+'.accum.txt', "w") as f:
        #Print header
        f.write("StartDate EndDate EnsembleName\n")

        #Print data from NumPy arrays
        for i in range(len(datstrt)):
            if i in np.arange(9):
                f.write(f"00{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")
            elif i in np.arange(9,99):
                f.write(f"0{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")
            else:
                f.write(f"{i+1} {str(datstrt[i]).strip('[]')} {str(datend[i]).strip('[]')} {enslist[i]}\n")
    fg.write_pickle(xtensi, './xtensi_CESM2-LE_'+exp+'_'+domainName+'_'+str(nDayAccum)+'.accum.pkl')
    fg.write_pickle(xtdati, './xtdati_CESM2-LE_'+exp+'_'+domainName+'_'+str(nDayAccum)+'.accum.pkl')

    return xtensi,xtdati