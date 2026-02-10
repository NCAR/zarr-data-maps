#!/usr/bin/env python
import xarray as xr
import xesmf as xe

import numpy as np

def get_latlon_b(ds,lon_str,lat_str,lon_dim,lat_dim):
    #### Longitude East-West Stagger
    diffWestEast    = ds[lon_str].diff(lon_dim)                         # take difference in longitudes across west-east (columns) dimension
    diffWestEast    = np.array(diffWestEast)                            # assign this to numpy array
    padding         = diffWestEast[:,-1].reshape(len(diffWestEast[:,-1]),1)  # get the last column of the differences and reshape so it can be appended
    diffWestEastAll = np.append(diffWestEast, padding, axis=1)/2               # append last value to all_data

    # dimensions now the same as ds
    lon_b_orig      = ds[lon_str]-diffWestEastAll

    # lon_b needs to be staggered - add to the final row to bound
    last_add     = ds[lon_str][:,-1]+diffWestEastAll[:,-1]
    last_add     = np.array(last_add)
    last_add     = last_add.reshape(len(last_add),1)
    lon_b_append = np.append(np.array(lon_b_orig),last_add,1)
    last_add     = lon_b_append[0,:].reshape(1,len(lon_b_append[0,:]))
    lon_b_append = np.append(last_add,lon_b_append,axis=0)


    #### Latitude Stagger
    diffSouthNorth=ds[lat_str].diff(lat_dim)                         # take difference in longitudes across west-east (columns) dimension
    diffSouthNorth=np.array(diffSouthNorth)                            # assign this to numpy array
    padding=diffSouthNorth[0,:].reshape(1,len(diffSouthNorth[0,:]))  # get the last column of the differences and reshape so it can be appended
    diffSouthNorthAll = np.append(padding,diffSouthNorth,axis=0)/2               # append last value to all_data

    # dimensions now the same as ds
    lat_b_orig      = ds[lat_str]-diffSouthNorthAll

    # lat_b needs to be staggered - add to the first row to bound
    last_add     = ds[lat_str][0,:]+diffWestEastAll[0,:]
    last_add     = np.array(last_add)
    last_add     = last_add.reshape(1,len(last_add))
    lat_b_append = np.append(last_add,np.array(lat_b_orig),axis=0)
    last_add     = lat_b_append[:,-1].reshape(len(lat_b_append[:,-1]),1)
    lat_b_append = np.append(lat_b_append,last_add,axis=1)


    grid_with_bounds = {'lon': ds[lon_str].values,
                               'lat': ds[lat_str].values,
                               'lon_b': lon_b_append,
                               'lat_b': lat_b_append,
                              }

    return grid_with_bounds

def get_latlon_b_rect(ds,lon_str,lat_str,lon_dim,lat_dim):
    #### Longitude Stagger
    diffWestEast    = ds[lon_str].diff(lon_dim)                         # take difference in longitudes across west-east (columns) dimension
    diffWestEastArr = np.array(diffWestEast).reshape(len(diffWestEast),1)                            # assign this to numpy array
    padding         = diffWestEastArr[-1].reshape(1,1)  # get the last column of the differences and reshape so it can be appended
    diffWestEastAll = np.append(diffWestEastArr, padding, axis=0)/2               # append last value to all_data
    # # dimensions now the same as ds
    lon_b_orig      = ds[lon_str].values.reshape(len(ds[lon_str]),1)-diffWestEastAll

    # lon_b needs to be staggered - add to the final row to bound
    last_add        = ds[lon_str][-1].values+diffWestEastAll[-1]
    last_add        = np.array(last_add).reshape(len(last_add),1)
    last_add        = last_add.reshape(1,1)
    lon_b_append    = np.append(np.array(lon_b_orig),last_add,0)

    #### Latitude Stagger
    diffSouthNorth    = ds[lat_str].diff(lat_dim)                         # take difference in latitudes across west-east (columns) dimension
    diffSouthNorthArr = np.array(diffSouthNorth).reshape(len(diffSouthNorth),1)                            # assign this to numpy array
    padding         = diffSouthNorthArr[-1].reshape(1,1)  # get the last column of the differences and reshape so it can be appended
    diffSouthNorthAll = np.append(diffSouthNorthArr, padding, axis=0)/2               # append last value to all_data
    # # dimensions now the same as ds
    lat_b_orig      = ds[lat_str].values.reshape(len(ds[lat_str]),1)-diffSouthNorthAll

    # lat_b needs to be staggered - add to the final row to bound
    last_add        = ds[lat_str][-1].values+diffSouthNorthAll[-1]
    last_add        = np.array(last_add).reshape(len(last_add),1)
    last_add        = last_add.reshape(1,1)
    lat_b_append    = np.append(np.array(lat_b_orig),last_add,0)

    grid_with_bounds = {'lon': ds[lon_str],
                               'lat': ds[lat_str],
                               'lon_b': lon_b_append.reshape(len(lon_b_append),),
                               'lat_b': lat_b_append.reshape(len(lat_b_append),),
                              }
    return grid_with_bounds


def main():
    # get_latlon_b_rect is for a rectilinear grid and get_latlon_b is for a curvilinear grid
    # Examples of calling them look like this:
    wrf_grid_with_bounds = get_latlon_b(dsWRF['PREC_ACC_NC'],lon_str='XLONG',lat_str='XLAT',lon_dim='west_east',lat_dim='south_north')
    # wrf_grid_with_bounds

    ICAR_grid_with_bounds = get_latlon_b(dsICAR['precipitation'],lon_str='lon',lat_str='lat',lon_dim='lon_x',lat_dim='lat_y')
    # Then regridding is as simple as
    regridder = xe.Regridder(wrf_grid_with_bounds, ICAR_grid_with_bounds, 'conservative') # input grid, gird you want to resample to, method
    dsWRF_out = regridder(dsWRF)


if __name__ == '__main__':
    main()
