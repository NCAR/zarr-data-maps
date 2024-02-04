import xarray as xr
import pandas as pd
import ndpyramid as ndp
import rioxarray

import sys
import zarr

if ndp.__version__ != '0.1.0':
    print(f"Error: ndpyramid version {ndp.__version__} != required 0.1.0")
    sys.exit(0)

VERSION = 2
LEVELS = 6
PIXELS_PER_TILE = 128

input_path = f"gs://carbonplan-maps/v{VERSION}/demo/raw"
save_path = f"data_demo"

input_path = '/glade/work/soren/src/icar/data/icar-zarr-data/data_input'
save_path = f"data_zarr/icar-noresm"

# input dataset
# only noresm_hist_exl_conv_2000_2005.nc

ds1 = []
# months = list(map(lambda d: d + 1, range(12)))
path = f"{input_path}/noresm_hist_exl_conv_2000_2005.nc"
print("Opening", path)
ds1 = xr.open_dataset(path, engine="netcdf4")


print("Selecting time frame")
ds1 = ds1.isel(time=slice(0, 12))

# Read icar_out files
# days = list(map(lambda d: d + 1, range(9,10)))
# for i in days:
#     path = f"{input_path}/icar_out_2000-{i:02g}-01_00-00-00.nc"  # tavg originally
#     # FOO: may need band variable??
#     ds = xr.open_dataset(path, engine="netcdf4")
#     if ds1:
#         ds1.append(ds)
#     else:
#         ds1 = ds
# #   older code
#    # .squeeze() #.reset_coords(["band"], drop=True)
#     # ds = (
#     #     xr.open_dataarray(path, engine="netcdf4") # this is dataset so open_dataset()
#     #      .to_dataset(name="climate")
#     #      .squeeze()
#     #      .reset_coords(["band"], drop=True)
#     # )



print("Transforming variables to match website")
# --- transform to dataset for website, test ---
# transform 2d lat lon dimension to 1d
lon_len = len(ds1.lat.shape)
if (lon_len == 1):
    pass
elif (lon_len == 2):
    ds1['lat'] = ds1.lat[:,0]
    ds1['lon'] = ds1.lon[0,:]
else:
    print("Not prepared to deal with lat/lon of dimension", lat_len)
    sys.exit(1)

# rename dimensions
ds1 = ds1.rename({'time':'month',
                  # 'lat_y':'y',
                  # 'lon_x':'x',
                  'lat':'y',
                  'lon':'x',
                  # 'precipitation':'prec',
                  'pcp':'prec',
                  # 'ta2m':'tavg'
                  't_mean':'tavg'
                  })
# add climate (aka variable) dimension
print(" - add climate (aka variable) dimension")
var1='prec'; var2='tavg'
ds1['climate'] = xr.concat([ds1[var1], ds1[var2]],
                           dim='band')

# cleanup variables
keep_vars = ['climate']
all_vars = list(ds1.data_vars)
remove_vars = [var for var in all_vars if var not in keep_vars]
ds1 = ds1.drop_vars(remove_vars)

# add band coordinates
print(" - add band coordinates")
band_var_names = ['prec','tavg']
fixed_length = 4
var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in band_var_names]
ds1 = ds1.assign_coords(band=var_names_U4)


# --- clean up types
print(" - clean up types")
# month to int type
ds1['month'] = xr.Variable(dims=('month',),
                           data=list(range(1, 12 + 1)))
                           # data=list(range(1, ds1.month.shape[0] + 1)))
                           # attrs={'dtype': 'int32'})
ds1["month"] = ds1["month"].astype("int32")
ds1["climate"] = ds1["climate"].astype("float32")
ds1["band"] = ds1["band"].astype("str")
ds1.attrs.clear()


# --- force to be like their data
print (" - force to be like their data[??]")
ds1 = ds1.where(ds1.month<=12, drop=True)


# sys.exit()


# --- create the pyramid
print("Create Pyramid")
# EPSG:4326 fixes error:
#    MissingCRS: CRS not found. Please set the CRS with 'rio.write_crs()'
ds1 = ds1.rio.write_crs('EPSG:4326')
dt = ndp.pyramid_reproject(ds1,
                           levels=LEVELS,
                           pixels_per_tile=PIXELS_PER_TILE,
                           extra_dim='band')

dt = ndp.utils.add_metadata_and_zarr_encoding(dt,
                                              levels=LEVELS,
                                              pixels_per_tile=PIXELS_PER_TILE)

print("Write to Zarr")
# write the pyramid to zarr, defaults to zarr_version 2
# consolidated=True, metadata files will have the information expected by site
dt.to_zarr(save_path + '/4d-ndp0.1/tavg-prec-month', consolidated=True)
