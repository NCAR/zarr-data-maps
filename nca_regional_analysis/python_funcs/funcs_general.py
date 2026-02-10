import xarray as xr
import xskillscore as xs
import numpy as np
import xesmf
import pandas as pd
import pyproj
import shapefile as shp
import matplotlib.path as mplPath
import pickle
# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import os
import glob


def load_wrf2d(diri, datstrt, datend, filotag='', diro=None, domtag='d01', COMPUTE=False, SAVE_DAILY=False, LOAD_DAILY=False, EVENT_TOTAL=False, LOAD_HOURLY=False, OVERWRITE=False):
    if diro is None:
        diro = diri
    filis = sorted(glob.glob(f'{diri}wrf2d_{domtag}*'))
    if COMPUTE:
        if os.path.exists(f'{diro}{filotag}_wrf2d_hourly.nc') and not OVERWRITE:
            print(f'File {diro}{filotag}_wrf2d_hourly.nc already exists. Set OVERWRITE=True or set COMPUTE=False to load existing file.')
            return
        else:
            FIRST = True
            for fil in filis:
                if FIRST:
                    ds = xr.open_dataset(fil,engine='netcdf4')
                    FIRST=False
                else:
                    tmp = xr.open_dataset(fil,engine='netcdf4')
                    ds = xr.concat([ds,tmp],dim='Time')
            ds = ds.rename({'Time':'time'})
            ds['time'] = (['time'], pd.date_range(start=datstrt, end=datend, freq='h'))
            ds['pr'] = (['time', 'south_north', 'west_east'], np.zeros(ds['RAINNC'].shape))
            for it in range(len(ds['time'])):
                if it==0:
                    tmpi = ds['RAINNC'][it,:,:].data + ds['SNOWNC'][it,:,:].data + ds['GRAUPELNC'][it,:,:].data
                    ds['pr'][it,:,:] = tmpi
                else:
                    tmpj = tmpi
                    tmpi = ds['RAINNC'][it,:,:].data + ds['SNOWNC'][it,:,:].data + ds['GRAUPELNC'][it,:,:].data
                    ds['pr'][it,:,:] = tmpi - tmpj
            if os.path.exists(f'{diro}{filotag}_wrf2d_hourly.nc'):
                os.remove(f'{diro}{filotag}_wrf2d_hourly.nc')
            ds.to_netcdf(f'{diro}{filotag}_wrf2d_hourly.nc')
    # elif EVENT_TOTAL:
    #     ds = xr.open_dataset(filis[-1],engine='netcdf4')
    #     return ds, None

            if SAVE_DAILY:
                dspr = ds[['pr']].resample(time='1D').sum()
                if os.path.exists(f'{diro}{filotag}_precip_day.nc'):
                    os.remove(f'{diro}{filotag}_precip_day.nc')
                dspr['XLONG'] = (['south_north','west_east'], ds['XLONG'][0,:,:].data)
                dspr['XLAT'] = (['south_north','west_east'], ds['XLAT'][0,:,:].data)
                dspr.to_netcdf(f'{diro}{filotag}_precip_day.nc')
                return ds, dspr
    if LOAD_DAILY:
        dspr = xr.open_dataset(f'{diro}{filotag}_precip_day.nc',engine='netcdf4')
        if LOAD_HOURLY:
            ds = xr.open_dataset(f'{diro}{filotag}_wrf2d_hourly.nc',engine='netcdf4')
            return ds, dspr
        else:
            return None, dspr
    # else:
    #     return ds, None


def get_month_length(year, month):
    if month == 2:
        if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            return 29
        else:
            return 28
    elif month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    else:
        return 30
    

def precipname(nci):
    if 'pcp' in list(nci.variables):
        prstr = 'pcp'
    elif 'pr' in list(nci.variables):
        prstr = 'pr'
    elif 'precipitation' in list(nci.variables):
        prstr = 'precipitation'
    elif 'precip' in list(nci.variables):
        prstr = 'precip'
    elif 'Prec' in list(nci.variables):
        prstr = 'Prec'
    return prstr


def read_pickle(fili):
    with open(fili, 'rb') as file:
        data = pickle.load(file)
    return data


def write_pickle(data,fili):
    with open(fili, 'wb') as file:
        pickle.dump(data, file)
    return


def GFG(arr,prec):
    new = np.array_str(arr, precision=prec, suppress_small=True)
    return new


def std_norm(test, verif):
    test_std = np.std(test)
    verif_std = np.std(verif)
    stdnorm = test_std/verif_std
    return stdnorm


def create_radius_mask(shape, center_idx, radius):
    y, x = center_idx
    y_indices, x_indices = np.ogrid[:shape[0], :shape[1]]
    distances = np.sqrt((y_indices - y)**2 + (x_indices - x)**2)
    mask = distances <= radius
    return mask


def compute_trend(nci,varname):
    month_length = nci.time.dt.days_in_month
    if varname == 'pr':
        # tmp = (nci['pr']*month_length)
        tmp = nci[varname]
        tmp = tmp.groupby('time.year').sum(dim='time',skipna=False)
    else:
        tmp = nci[varname]
        tmp = tmp.groupby('time.year').mean(dim='time',skipna=False)
    trend = xs.linslope(tmp['year'].astype(float),tmp,dim='year',skipna=True)
    trend = trend*10.
    if varname == 'pr':
        trend.attrs['units'] = 'mm/decade'
    elif varname in ['tas','tasmax','tasmin']:
        trend.attrs['units'] = '°C/decade'
    else:
        trend.attrs['units'] = 'units/decade'
    return trend


def coordnames(nci):
    if 'latitude' in list(nci.variables):
        latstr = 'latitude'
    elif 'lat' in list(nci.variables):
        latstr = 'lat'
    elif 'nav_lat' in list(nci.variables):
        latstr = 'nav_lat'

    if 'longitude' in list(nci.variables):
        lonstr = 'longitude'
    elif 'lon' in list(nci.variables):
        lonstr = 'lon'
    elif 'nav_lon' in list(nci.variables):
        lonstr = 'nav_lon'

    if 'lat' in list(nci.dims):
        latdim = 'lat'    
        londim = 'lon'
    elif 'latitude' in list(nci.dims):
        latdim = 'latitude'
        londim = 'longitude'
    elif 'nav_lat' in list(nci.dims):
        latdim = 'nav_lat'
        londim = 'nav_lon'
    elif 'x' in list(nci.dims):
        latdim = 'y'
        londim = 'x'
    else:
        latdim = 'j'
        londim = 'i'
    return latstr,lonstr,latdim,londim


def fixlons(nci,latdim,londim,lonstr):
    lonarray = np.zeros(nci[lonstr].shape)
    ndim = lonarray.ndim
    if ndim == 2:
        if float(nci[lonstr].min()) < -1.:
            for i in range(nci[lonstr].shape[0]):
                for j in range(nci[lonstr].shape[1]):
                    if float(nci[lonstr][i,j]) < 0.:
                        lonarray[i,j] = nci[lonstr][i,j].data + 360.
                    else:
                        lonarray[i,j] = nci[lonstr][i,j].data
            nci[lonstr] = ([latdim, londim], lonarray)
    elif ndim == 1:
        if float(nci[lonstr].min()) < -1.:
            for i in range(nci[lonstr].shape[0]):
                if float(nci[lonstr][i]) < 0.:
                    lonarray[i] = nci[lonstr][i].data + 360.
                else:
                    lonarray[i] = nci[lonstr][i].data
            nci[lonstr] = ([londim], lonarray)
    return nci


def find_latlon_edges(data):
    """
    Finds edge lat/lon of data
    Assumes 1D lat/lon
    """
    lonmin = data['lon'][0]
    lonmax = data['lon'][-1]
    lonslic = slice(lonmin,lonmax)
    latmin = data['lat'][0]
    latmax = data['lat'][-1]
    latslic = slice(latmin,latmax)
    return lonmin, lonmax, lonslic, latmin, latmax, latslic


def regrid_with_nan(data,regridder, C=10.):
    """
    Regrids data using XESMF regridder while masking NaN values
    (Default XESMF regridder replaces NaN with 0)
    """
    data = data + C
    data_rg = regridder(data)
    data_rg = xr.where(data_rg == 0.0, np.nan, data_rg)
    # data_rg[data_rg==0.0] = np.nan
    data_rg = data_rg - C
    return data_rg


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def get_buco_mask(targetds, bucods, region):
    if targetds['lon'].ndim == 1:
        lonmin = targetds['lon'][0]
        lonmax = targetds['lon'][-1]

        latmin = targetds['lat'][0]
        latmax = targetds['lat'][-1]
    else:
        lonmin = targetds['lon'][0,0]
        lonmax = targetds['lon'][-1,-1]

        latmin = targetds['lat'][0,0]
        latmax = targetds['lat'][-1,-1]
    lonslic = slice(lonmin,lonmax)
    latslic = slice(latmin,latmax)
    bucotmp = bucods.sel(lat=latslic,lon=lonslic)
    bucorgr = xesmf.Regridder(bucotmp,targetds,method='nearest_s2d')
    bucotmp = bucorgr(bucotmp['mask'])
    bucomask = xr.where(np.isnan(targetds),np.nan,bucotmp).squeeze()
    if region == 'Desert Southwest':
        bucomask = xr.where((bucomask['lat']>30)  &
                           (bucomask['lat']<40)  & 
                           (bucomask['lon']>240) & 
                           (bucomask['lon']<255.5),bucomask,np.nan).squeeze()
    return bucomask


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
            ctr=np.array(ctr)[::100,:] # carsen the shapefile accuracy
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
        sShapefiles = '/glade/u/home/nlybarger/shapefiles/BOR_Contributing_Area.shp'
        prj_file_path = sShapefiles.replace('.shp', '.prj')
        sf = shp.Reader(sShapefiles)
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


def mapper(array, lon, lat, vmin=0, vmax=1, title='', cm='seismic', titfontsize=12, lontickfreq=10., lattickfreq=10.):
    """
    Plot array or DataArray on Mercator projection using Cartopy.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.Mercator()})
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

    px, py = np.meshgrid(lon, lat)
    mesh = ax.pcolormesh(px, py, array, vmin=vmin, vmax=vmax, cmap=cm, transform=ccrs.PlateCarree())

    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    # Set gridlines and labels
    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(int(lon[0]), int(lon[-1])+1, lontickfreq),
                      ylocs=np.arange(int(lat[0]), int(lat[-1])+1, lattickfreq),
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title(title, fontsize=titfontsize)
    plt.colorbar(mesh, ax=ax, orientation='vertical', extend='neither', pad=0.02, aspect=30)
    return ax


def compute_IWV_IVT(u,v,q,coord_vert, levunit='Pa'):
    """
    Computes IWV and IVT from u,v,q DataArrays
    Must specify vertical coordinate
    u,v,q must all have same dimensions
    """
    g=9.81
    if levunit=='hPa':
        u[coord_vert] = u[coord_vert]*100.
        v[coord_vert] = v[coord_vert]*100.
        q[coord_vert] = q[coord_vert]*100.
    UV = np.sqrt(np.square(u) + np.square(v))
    dp = UV[coord_vert].diff(coord_vert)
    dp = dp.reindex({coord_vert: UV[coord_vert]})
    IWV = ((1/g)*q*dp).sum(coord_vert)
    IVT = ((1/g)*UV*q*dp).sum(coord_vert)
    return IWV,IVT


def seasonal_avg_vars(nci,latstr,lonstr,ELI=False):
    years = list(nci.groupby('time.year').groups)
    nyr = len(years)
    drs = {}
    seaskeys=['DJF','MAM','JJA','SON','ANN']
    for seas in seaskeys:
        drs[seas] = {}
    for iy in range(nyr-1):
        drs['DJF'][iy] = slice(str(years[iy])+'-12-01',str(years[iy+1])+'-02-28')
        drs['MAM'][iy] = slice(str(years[iy+1])+'-03-01',str(years[iy+1])+'-05-30')
        drs['JJA'][iy] = slice(str(years[iy+1])+'-06-01',str(years[iy+1])+'-08-30')
        drs['SON'][iy] = slice(str(years[iy+1])+'-09-01',str(years[iy+1])+'-11-30')
        drs['ANN'][iy] = slice(str(years[iy+1])+'-01-01',str(years[iy+1])+'-12-31')
    nci['pranom'] = nci['pr'].groupby('time.month') - nci['pr'].groupby('time.month').mean(dim='time')
    
    datout = {}
    outvars = {}
    nci['tasanom'] = nci['tas'].groupby('time.month') - nci['tas'].groupby('time.month').mean(dim='time')
    outvars['tas'] = {}
    outvars['tasanom'] = {}
    if ELI:
        outvars['eli'] = {}
    outvars['pr'] = {}
    outvars['pranom'] = {}
    outvars['n34'] = {}

    for seas in seaskeys:
        outvars['n34'][seas] = np.zeros(nyr-1)
        outvars['pr'][seas]  = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        outvars['pranom'][seas]  = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        outvars['tas'][seas] = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        outvars['tasanom'][seas] = np.zeros((nyr-1,len(nci[latstr]),len(nci[lonstr])))
        if ELI:
            outvars['eli'][seas] = np.zeros(nyr-1)

        for iy in range(nyr-1):
            outvars['n34'][seas][iy]           = nci['n34'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pranom'][seas][iy,:,:]    = nci['pranom'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['pr'][seas][iy,:,:]        = nci['pr'].sel(time=drs[seas][iy]).sum(dim='time').values
            outvars['tas'][seas][iy,:,:]       = nci['tas'].sel(time=drs[seas][iy]).mean(dim='time').values
            outvars['tasanom'][seas][iy,:,:]   = nci['tasanom'].sel(time=drs[seas][iy]).mean(dim='time').values
            if ELI:
                outvars['eli'][seas][iy]           = nci['eli'].sel(time=drs[seas][iy]).mean(dim='time').values
        
        if nci['lat'].ndim == 1:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['time'], outvars['n34'][seas]),
                    pr=(['time','lat','lon'],outvars['pr'][seas]),
                    pranom=(['time','lat','lon'],outvars['pranom'][seas]),
                    tas=(['time','lat','lon'],outvars['tas'][seas]),
                    tasanom=(['time','lat','lon'],outvars['tasanom'][seas]),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    lat = (['lat'],nci[latstr].data),
                    lon = (['lon'],nci[lonstr].data),
                ),
            )
        else:
            datout[seas] = xr.Dataset(
                data_vars = dict(
                    n34=(['time'], outvars['n34'][seas]),
                    pr=(['time','y','x'],outvars['pr'][seas]),
                    pranom=(['time','y','x'],outvars['pranom'][seas]),
                    tas=(['time','y','x'],outvars['tas'][seas]),
                    tasanom=(['time','y','x'],outvars['tasanom'][seas]),
                    lat=(['y','x'],nci['lat'].data),
                    lon=(['y','x'],nci['lon'].data),
                ),
                coords = dict(
                    time = pd.date_range(str(years[0]+1)+'-01-01', periods=nyr-1, freq='YS'),
                    y = (['y'],nci['y'].data),
                    x = (['x'],nci['x'].data),
                ),
            )
        if ELI:
            datout[seas]['eli'] = (['time'],outvars['eli'][seas])
    return datout


def compute_rh_from_qt(q, T, P, outname=None):
    """
    Compute relative humidity (RH) from specific humidity (q), temperature (T), and pressure (P).
    
    Parameters:
    q (float, ndarray, or xarray.DataArray): Specific humidity in kg/kg
    T (float, ndarray, or xarray.DataArray): Temperature in Kelvin
    P (float, ndarray, or xarray.DataArray): Pressure in hPa
    outname (string): Name of output variable xr.Dataset (default is None)
    
    Returns:
    float, ndarray, or xarray.DataArray: Relative humidity in percentage
    """
    # Constants
    Rv = 461.5  # Specific gas constant for water vapor (J/(kg·K))
    Rd = 287.05 # Specific gas constant for dry air (J/(kg·K))
    epsilon = Rd / Rv
    
    # Calculate saturation vapor pressure using Tetens formula
    # Check if T has units attribute and is in Kelvin, convert to Celsius if so
    if hasattr(T, 'attrs') and 'units' in T.attrs:
        if T.attrs['units'].lower() in ['k', 'kelvin']:
            T_c = T - 273.15
    elif np.nanmean(T) > 100:  # crude check for Kelvin values
        T_c = T - 273.15
    else:
        T_c = T  # Assume already in Celsius
    es = (np.exp((34.494 - (4924.99/(T_c+237.1)))) / ((T_c + 105.)**1.57)) / 100 # Convert Pa to hPa
    
    # Calculate actual vapor pressure
    e = (q * P) / (epsilon + (1 - epsilon) * q)
    
    # Calculate relative humidity
    RH = (e / es) * 100
    
    # If input is xarray.DataArray, preserve attributes and coordinates
    if isinstance(q, xr.DataArray):
        RH = xr.DataArray(RH, dims=q.dims, coords=q.coords)
        RH.attrs['units'] = '%'
        RH.attrs['long_name'] = 'Relative Humidity at pressure surface'
        RH.attrs['cell_methods'] = ['time: mean']
        # RH=RH.to_dataset(name='RH' if outname is None else outname)
    
    return RH

def compute_Q_from_rh(RH, T, P, outname=None):
    """
    Compute specific humidity (q) from relative humidity (RH), temperature (T), and pressure (P).
    
    Parameters:
    RH (float, ndarray, or xarray.DataArray): Relative humidity in percentage
    T (float, ndarray, or xarray.DataArray): Temperature in Kelvin
    P (float, ndarray, or xarray.DataArray): Pressure in hPa
    outname (string): Name of output variable xr.Dataset (default is None)
    
    Returns:
    float, ndarray, or xarray.DataArray: Specific humidity in kg/kg
    """
    # Constants
    Rv = 461.5  # Specific gas constant for water vapor (J/(kg·K))
    Rd = 287.05 # Specific gas constant for dry air (J/(kg·K))
    epsilon = Rd / Rv
    
    # Calculate saturation vapor pressure using Tetens formula
    # Check if T has units attribute and is in Kelvin, convert to Celsius if so
    if hasattr(T, 'attrs') and 'units' in T.attrs:
        if T.attrs['units'].lower() in ['k', 'kelvin']:
            T_c = T - 273.15
    elif np.nanmean(T) > 100:  # crude check for Kelvin values
        T_c = T - 273.15
    else:
        T_c = T  # Assume already in Celsius
    es = (np.exp((34.494 - (4924.99/(T_c+237.1)))) / ((T_c + 105.)**1.57)) / 100 # Convert Pa to hPa
    
    # Calculate actual vapor pressure
    e = (RH / 100) * es
    
    # Calculate specific humidity
    q = (epsilon * e) / (P - (1 - epsilon) * e)
    
    # If input is xarray.DataArray, preserve attributes and coordinates
    if isinstance(RH, xr.DataArray):
        q = xr.DataArray(q, dims=RH.dims, coords=RH.coords)
        q.attrs['units'] = 'kg/kg'
        q.attrs['long_name'] = 'Specific Humidity at pressure surface'
        q.attrs['cell_methods'] = ['time: mean']
        if outname is not None:
            q=q.to_dataset(name=outname)
    
    return q

def compute_spi_xarray(precip, scale):
    """
    Compute the Standardized Precipitation Index (SPI) for a given time scale using xarray.

    Parameters:
    precip (xarray.DataArray): Monthly precipitation data (time, lat, lon).
    scale (int): Time scale in months (e.g., 12 for 1 year, 24 for 2 years, 60 for 5 years).

    Returns:
    xarray.DataArray: SPI values for the given time scale.
    """
    # Step 1: Compute rolling sum over the specified scale
    rolling_precip = precip.rolling(time=scale, center=True, min_periods=1).mean()
    # Step 2: Define a function to compute SPI for a single time series
    def calculate_spi(ts):
        spi = np.full_like(ts, np.nan)
        valid_idx = ~np.isnan(ts) & (ts > 0) & np.isfinite(ts)
        # Require at least 10 valid points for fitting (adjust as needed)
        if valid_idx.sum() > 10:
            try:
                shape, loc, scale_param = gamma.fit(ts[valid_idx], floc=0)
                cdf = gamma.cdf(ts[valid_idx], shape, loc=loc, scale=scale_param)
                # Avoid cdf=0 or 1 for norm.ppf
                cdf = np.clip(cdf, 1e-10, 1-1e-10)
                spi[valid_idx] = norm.ppf(cdf)
            except Exception:
                spi[:] = np.nan
        return spi


    spi = xr.apply_ufunc(
        calculate_spi,
        rolling_precip,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask="parallelized",  # Enable parallel computation if using Dask
        output_dtypes=[float],
    )
    spi = spi.transpose(*precip.dims)  # Transpose to match input dimensions
    return spi