import argparse
import os
import pandas as pd
import ndpyramid as ndp
import rioxarray
import sys
import xarray as xr
import xesmf as xe
import zarr
from tools import dimensionNames, handleArgs

print("ndpyramid Version = ", ndp.__version__)
# sys.exit()
# --- combination of downscaling methods and climate models to create data
# old
# downscaling_methods = [
#     'icar',
#     'gard',
#     'LOCA',
#     'bcsd',]
# climate_models = [
#     'noresm',
#     'cesm',
#     'gfdl',
#     'miroc5',]
# new
# [model].[method].ds.conus.metric.maps.nc
downscaling_methods = [
    'ICAR',
    # 'ICARwest',
    'GARD_r2',
    'GARD_r3',
    # 'GARDwest',
    'LOCA_8th',
    'MACA',
    'NASA-NEX',]
climate_models = [
    'ACCESS1-3',
    'CanESM2',
    'CCSM4',
    'MIROC5',
    'NorESM1-M',]
observation_datasets = [
    'CONUS404',
    'GMET',
    'Livneh',
    'Maurer',
    'NLDAS',
    'oldLivenh',
    'PRISM',]

# set these somewhere else?
VERSION = 2
LEVELS = 6
PIXELS_PER_TILE = 256
time = 'time'
time_slices=[slice("1980","2010"), slice("2070","2100")]
time_slice_strs=['1980_2010', '2070_2100']
# time_slice=slice("1980","2010")
# time_slice_str='1980_2010'
time_slice=slice("2070","2100")
time_slice_str='2070_2100'

# testing new
time_slice_str='1980_2010'

time_slice_str='1981_2004'

PAST=0
FUTURE=1
CLIMATE_SIGNAL=2


class Dataset:
    def __init__(self, method, model, past=None, future=None, rcp=''):
        if (past != None) and not os.path.exists(past):
            print("ERROR: past path does not exist:", past)
            sys.exit()
        if (future != None) and not os.path.exists(future):
            print("ERROR: future path does not exist:", future)
            sys.exit()
        self.past_path = past
        self.future_path = future
        self.method = method
        self.model = model
        self.rcp = rcp
    def print(self):
        print("Dataset:")
        print("  past_path =", self.past_path)
        print("  future_path =", self.future_path)
        print("  method =", self.method)
        print("  model =", self.model)



class Options:
    def __init__(self, input_path, past_path=None, future_path=None,
                 climate_signal_path=None):
        self.input_path = input_path
        self.write_past = False
        self.past_path = None
        self.write_future = False
        self.future_path = None
        self.write_climate_signal = False
        self.climate_signal_path = None
        if past_path != None:
            self.write_past = True
            self.past_path = self.check_path(past_path)
        if future_path != None:
            self.write_future = True
            self.future_path = self.check_path(future_path)
        if climate_signal_path != None:
            self.write_climate_signal = True
            self.climate_signal_path = self.check_path(climate_signal_path)
    def check_path(self, path):
        if isinstance(path, list):
            path = path[0]
        if path[-1] != '/':
            path += '/'
        return path
    def print(self):
        print("Options:")
        print("  input path =", self.input_path)
        print("  past write =", self.write_past,
              ", path =", self.past_path)
        print("  future write =", self.write_future,
              ", path =", self.future_path)
        print("  climate signal write =", self.write_climate_signal,
              ", path =", self.climate_signal_path)


def create_comparison_combinations(comparison_paths):
    method_combinations = itertools.product(downscaling_methods, repeat=2)
    method_paths = ['_'.join(items) for items in method_combinations]
    model_combinations = itertools.product(climate_models, repeat=2)
    model_paths = ['_'.join(items) for items in model_combinations]
    time_combinations = itertools.product(time_slices, repeat=2)
    time_paths = ['_'.join(items) for items in time_combinations]
    all_combinations = itertools.product(method_paths, model_paths, time_paths)
    comparison_path_combinations = ['/'.join(items) for items in all_combinations]
    return comparison_path_combinations

def comparisons():
    print("--- Starting Zarr Comparison Data Maps Setup ---")
    library_check()
    downscaling_methods = ['icar'] # !!!! TESTING ONLY, REMOVE LATER !!!!
    comparison_paths = handleArgs.get_comparison_arguments(downscaling_methods,
                                                           climate_models,
                                                           time_slice_strs)
    comparison_path_combinations = create_comparison_combinations(comparison_paths)

    var_name = 'climate'
    count = 0
    for comparisons in comparison_path_combinations:
        (z1_input_path, z2_input_path) = comparisons
        z1 = zarr.open_group(z1_input_path, mode='r')
        z2 = zarr.open_group(z2_input_path, mode='r')
        zdiff = zarr.open_group('data/example.zarr', mode='a')

        groups = []
        for name, value in z1.groups():
            groups.append(name + '/' + var_name)


        break

        count += 1
        if (count == 2):
            break

    print('Fin')
# {model}.{method}.ds.conus.metric.maps.nc

def handlePastFutureArgs(input_path, time_period):
    datasets = []

    if time_period == PAST:
        rcps = ['']
    elif time_period == FUTURE:
        rcps = ['.rcp45', '.rcp85']

    for cm in climate_models:
        for dm in downscaling_methods:
            for rcp in rcps:
                filepath = input_path+'/'+cm+'.'+dm+rcp+'.ds.conus.metric.maps.nc'
                if (time_period == PAST and os.path.exists(filepath)):
                    datasets.append(Dataset(dm, cm, past=filepath, rcp=rcp))
                elif (time_period == FUTURE and os.path.exists(filepath)):
                    datasets.append(Dataset(dm, cm, future=filepath, rcp=rcp))
    return datasets


def handleClimateSignalArgs(input_path):
    rcps = ['rcp45', 'rcp85']
    datasets = []

    for cm in climate_models:
        for dm in downscaling_methods:
            past_path = input_path+'/'+cm+'.'+dm+'.ds.conus.metric.maps.nc'
            for rcp in rcps:
                future_path = input_path+'/'+cm+'.'+dm+'.'+rcp+'.ds.conus.metric.maps.nc'
                if os.path.exists(past_path) and os.path.exists(future_path):
                    datasets.append(Dataset(dm, cm, past_path, future_path, rcp=rcp))
    return datasets



def handleObsArgs(observations):
    path = sys.argv[1]
    paths = []
    for ob in observations:
        filepath = path+'/'+ob+'.ds.conus.metric.maps.nc'
        paths.append(filepath)
    return paths

def process_obs():
    print("--- Starting Zarr Data Maps Setup ---")
    library_check()
    paths = handleObsArgs(observation_datasets)

    print(paths)
    for i,path in enumerate(paths):
        # print(path)
        if not os.path.exists(path):
            print("does not exist:", path)
            continue
        ob = observation_datasets[i]
        print(i, ":CHECK: ",ob.lower().replace('-','_'))
        ds = xr.open_dataset(path)

        # for a in ds.variables.keys():
        #     print(a)
        # sys.exit()

        # rename dimensions
        # 'lat_y': y_name,
        # 'lon_x': x_name,
        new_dims = {#'time': 'time',
                    'lat': 'y',
                    'lon': 'x',
                    'n34pr':'n34p',
                    'ttrend':'ttre',
                    'ptrend':'ptre',
                    'pr90':'pr90',
                    'pr99':'pr99',
                    't90':'t90_',
                    't99':'t99_',
                    'djf_t':'djft',
                    'djf_p':'djfp',
                    'mam_t':'mamt',
                    'mam_p':'mamp',
                    'jja_t':'jjat',
                    'jja_p':'jjap',
                    'son_t':'sont',
                    'son_p':'sonp',
        }
        ds = ds.rename(new_dims)
        variables = list(ds.variables.keys())
        variables = [var for var in variables if var not in ['x', 'y']]
        # print(variables)
        # sys.exit()
        # don't need to average and handle time dimension
        # - [ ] do i need to handle time?
        # don't need to drop extra dimensions
        # add climate dimension
        # add climate (aka variable) dimension
        print(" - add climate (aka variable) dimension")
        # variables = ['prec', 'tavg']
        fixed_length = 4
        concatenated_vars = []
        for var_name in variables:
            concatenated_vars.append(ds[var_name])
        ds['climate'] = xr.concat(concatenated_vars, dim='band')
        var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
        ds = ds.assign_coords(band=var_names_U4)
        ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
        # --- clean up types
        print(" - clean up types")
        # month to int type
        ds['month'] = xr.Variable(dims=('month',),
                           data=list(range(1, 12 + 1)))
                           # data=list(range(1, ds.month.shape[0] + 1)))
                           # attrs={'dtype': 'int32'})
        ds["month"] = ds["month"].astype("int32")
        ds["climate"] = ds["climate"].astype("float32")
        ds["band"] = ds["band"].astype("str")
        ds.attrs.clear()
        ds = convert_to_zarr_format(ds) # already in single precision

        write_obs_to_zarr(ds, ob.lower().replace('-','_'))

        print(ds)
        # sys.exit()

    print('---fin---')
    sys.exit()


# parse command line arguments
def parseCLA():
    # define argument parser
    parser = argparse.ArgumentParser(description="Create Zarr files for ICAR Maps.")
    parser.add_argument("input_path", help="Input file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose mode")
    group = parser.add_argument_group('Output Data', 'types of output and path to  write to')
    group.add_argument("--past", nargs=1, dest="past_path",
                       help="Write past model data to passed path")
    group.add_argument("--future", nargs=1, dest="future_path",
                       help="Write future model data to passed path")
    group.add_argument("--climate-signal", nargs=1, dest="climate_signal_path",
                       help="Write climate signal data to passed path")

    # Parse the arguments
    args = parser.parse_args()

    if (args.past_path == None and
        args.future_path == None and
        args.climate_signal_path == None):
        print("ERROR: past, future, or climate-signal options are required")
        print(parser.print_help())
        sys.exit()

    options = Options(args.input_path, args.past_path, args.future_path,
                      args.climate_signal_path)

    return options

def main():
    print("--- Starting Zarr Data Maps Setup ---")
    library_check()

    # parse command line arguments
    options = parseCLA()

    # organize this better in the future
    # process_obs()

    # paths, method, model = handleArgs.get_arguments(downscaling_methods, climate_models)
    # paths, methods, models = new_handleArgs(input_path, time_period)
    options.print()
    print('---')
    if options.write_past:
        past_datasets = handlePastFutureArgs(options.input_path, PAST)
    if options.write_future:
        future_datasets = handlePastFutureArgs(options.input_path, FUTURE)
    if options.write_climate_signal:
        climate_signal_datasets = handleClimateSignalArgs(options.input_path)

    sys.exit()
    count = 0
    for dataset in datasets:
        count+=1
        method = dataset.method
        model = dataset.model
        # print(":CHECK: ",method.lower().replace('-','_'), model.lower().replace('-','_'))
        # print(":Opened", method, "downscaling method and", model, "climate model")
        ds_past = xr.open_dataset(dataset.past_path)
        ds_future = xr.open_dataset(dataset.future_path)
        ds = ds_future - ds_past

        new_dims = {#'time': 'time',
            'lat': 'y',
            'lon': 'x',
            'n34pr':'n34p',
            'ttrend':'ttre',
            'ptrend':'ptre',
            'pr90':'pr90',
            'pr99':'pr99',
            't90':'t90_',
            't99':'t99_',
            'djf_t':'djft',
            'djf_p':'djfp',
            'mam_t':'mamt',
            'mam_p':'mamp',
            'jja_t':'jjat',
            'jja_p':'jjap',
            'son_t':'sont',
            'son_p':'sonp',
        }
        ds = ds.rename(new_dims)
        # print("ds after rename =", ds)
        variables = list(ds.variables.keys())
        variables = [var for var in variables if var not in ['x', 'y']]

        # print(" - add climate (aka variable) dimension")
        # variables = ['prec', 'tavg']
        fixed_length = 4
        concatenated_vars = []
        for var_name in variables:
            concatenated_vars.append(ds[var_name])
        ds['climate'] = xr.concat(concatenated_vars, dim='band')
        var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
        ds = ds.assign_coords(band=var_names_U4)
        ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
        # --- clean up types
        # print(" - clean up types")
        # month to int type
        # ds['month'] = xr.Variable(dims=('month',),
        #                    data=list(range(1, 12 + 1)))
        #                    # data=list(range(1, ds.month.shape[0] + 1)))
        #                    # attrs={'dtype': 'int32'})
        # ds["month"] = ds["month"].astype("int32")
        ds["climate"] = ds["climate"].astype("float32")
        ds["band"] = ds["band"].astype("str")
        ds.attrs.clear()
        ds = convert_to_zarr_format(ds) # already in single precision

        write_to_zarr(ds,
                      output_path,
                      method.lower().replace('-','_'),
                      model.lower().replace('-','_'),
                      future = True,
                      rcp = dataset.rcp)
        print("small fin", 'output_path=',output_path)
        # if (count == 4):
        #     sys.exit()

    print('---fin---')
    sys.exit()



    for i,path in enumerate(paths):
        # print(path)
        # if i < 3:
        #     continue
        if not os.path.exists(path):
            print("does not exist:", path)
            continue
        method = methods[i]
        model = models[i]
        print(i, ":CHECK: ",method.lower().replace('-','_'), model.lower().replace('-','_'))
        print(i, ":Opened", method, "downscaling method and", model, "climate model")
        ds = xr.open_dataset(path)

        print("DS1 =", ds)
        # rename dimensions
        # 'lat_y': y_name,
        # 'lon_x': x_name,
        new_dims = {#'time': 'time',
                    'lat': 'y',
                    'lon': 'x',
                    'n34pr':'n34p',
                    'ttrend':'ttre',
                    'ptrend':'ptre',
                    'pr90':'pr90',
                    'pr99':'pr99',
                    't90':'t90_',
                    't99':'t99_',
                    'djf_t':'djft',
                    'djf_p':'djfp',
                    'mam_t':'mamt',
                    'mam_p':'mamp',
                    'jja_t':'jjat',
                    'jja_p':'jjap',
                    'son_t':'sont',
                    'son_p':'sonp',
        }
        ds = ds.rename(new_dims)
        print("ds after rename =", ds)
        variables = list(ds.variables.keys())
        variables = [var for var in variables if var not in ['x', 'y']]
        # print(variables)
        # sys.exit()
        # don't need to average and handle time dimension
        # - [ ] do i need to handle time?
        # don't need to drop extra dimensions
        # add climate dimension
        # add climate (aka variable) dimension
        print(" - add climate (aka variable) dimension")
        # variables = ['prec', 'tavg']
        fixed_length = 4
        concatenated_vars = []
        for var_name in variables:
            concatenated_vars.append(ds[var_name])
        ds['climate'] = xr.concat(concatenated_vars, dim='band')
        var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
        ds = ds.assign_coords(band=var_names_U4)
        ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
        # --- clean up types
        print(" - clean up types")
        # month to int type
        # ds['month'] = xr.Variable(dims=('month',),
        #                    data=list(range(1, 12 + 1)))
        #                    # data=list(range(1, ds.month.shape[0] + 1)))
        #                    # attrs={'dtype': 'int32'})
        # ds["month"] = ds["month"].astype("int32")
        ds["climate"] = ds["climate"].astype("float32")
        ds["band"] = ds["band"].astype("str")
        ds.attrs.clear()
        ds = convert_to_zarr_format(ds) # already in single precision

        print("From input file", path)
        write_to_zarr(ds, output_path, method.lower().replace('-','_'), model.lower().replace('-','_'))


        print("small fin")
        # sys.exit()

    print('---fin---')
    sys.exit()
    ds = open_data_srcs(paths)
    print("Opened", method, "downscaling method and", model, "climate model")

    ds = rename_dimensions(ds, method, model)
    ds = average_and_handle_time_dimension(ds)

    ds = drop_extra_dimensions(ds)
    ds = add_climate_dimension(ds)

    ds = convert_to_zarr_format(ds) # already in single precision
    write_to_zarr(ds, method, model)

    print('fin')
    sys.exit()


# library check
def library_check():
    # if ndp.__version__ != '0.1.0':
    #     print(f"Error: ndpyramid version {ndp.__version__} != required 0.1.0")
    #     sys.exit(0)
    return


def open_data_srcs(data_srcs):
    print("OPENING ", data_srcs)
    if (len(data_srcs) == 1):
        ds = xr.open_dataset(data_srcs[0])
    else:
        datasets = []
        for f in data_srcs:
            datasets.append(xr.open_dataset(f))
        ds = xr.concat(datasets, dim='time')
    return ds

def rename_dimensions(ds, method, model):
    new_dims = dimensionNames.get_dimension_name(method, model)
    ds = ds.rename(new_dims)
    return ds

def drop_extra_dimensions(ds):
    vars_to_keep = [time, 'y', 'x', 'prec', 'tavg']
    vars_to_drop = [var for var in ds.variables if var not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)
    return ds

def average_and_handle_time_dimension(ds):
    print("Creating spatial and monthly average")
    print("   - if silent failure, run on interactive node")

    # --- smaller time slice for testing
    # print("REMOVE 1999-2001 TIME SLICE")
    ds = ds.sel(time=time_slice)

    ds_subset = ds[['tavg', 'prec']]
    monthly_avg = ds_subset.resample(time='MS').mean(dim='time')
    monthly_avg_across_years = monthly_avg.groupby('time.month').mean(dim='time')
    return monthly_avg_across_years
    # spatial_avg_precip = ds['prec'].mean(dim=['y', 'x'])
    # spatial_avg_temp = ds['tavg'].mean(dim=['y', 'x'])
    # monthly_avg_precip = spatial_avg_precip.resample(time='MS').mean(dim='time')
    # monthly_avg_temp = spatial_avg_temp.resample(time='MS').mean(dim='time')
    # avg_precip = ds.resample(time='MS').mean(dim='time')
    # avg_temp = ds.resample(time='MS').mean(dim='time')


    # ds = xr.Dataset({'prec': avg_precip,
    #                  'tavg': avg_temp
    # #                   # 't_min': monthly_min_temp,
    # #                   # 't_max': monthly_max_temp
    #               })

    # return ds

# def write_to_zarr(ds, method, model):
#     print("ARTLESS CLEAN THIS UP!!")
#     print("Writing to zarr format")
#     save_path = f'data/chart/' + method + '/' + model
#     prec_save_path = save_path + '/prec'
#     tavg_save_path = save_path + '/tavg'

#     print("  WARNING: NEED TO ADD LARGER TIME FRAME")
#     # write precip
#     z_prec = zarr.open(prec_save_path, mode='w', shape=len(ds.prec),
#                   compressor=None, dtype=np.float32)
#     z_prec[:] = ds.prec.data

#     # write temp
#     z_temp = zarr.open(tavg_save_path, mode='w', shape=len(ds.tavg),
#                   compressor=None, dtype=np.float32)
#     z_temp[:] = ds.tavg.data

#     # write attributes
#     # z_prec.attrs['TIME START'] = '1980'
#     # z_temp.attrs['TIME START'] = '1970'



# months = list(map(lambda d: d + 1, range(12)))
# path = f"{input_path}/noresm_hist_exl_conv_2000_2005.nc"
# print("Opening", path)
# ds1 = xr.open_dataset(path, engine="netcdf4")

# print("Selecting time frame")
# ds1 = ds1.isel(time=slice(0, 12))

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



# print("Transforming variables to match website")
# # --- transform to dataset for website, test ---
# # transform 2d lat lon dimension to 1d
# lon_len = len(ds1.lat.shape)
# if (lon_len == 1):
#     pass
# elif (lon_len == 2):
#     ds1['lat'] = ds1.lat[:,0]
#     ds1['lon'] = ds1.lon[0,:]
# else:
#     print("Not prepared to deal with lat/lon of dimension", lat_len)
#     sys.exit(1)

# # rename dimensions
# ds1 = ds1.rename({'time':'month',
#                   # 'lat_y':'y',
#                   # 'lon_x':'x',
#                   'lat':'y',
#                   'lon':'x',
#                   # 'precipitation':'prec',
#                   'pcp':'prec',
#                   # 'ta2m':'tavg'
#                   't_mean':'tavg'
#                   })

def add_climate_dimension(ds):
    # add climate (aka variable) dimension
    print(" - add climate (aka variable) dimension")


    variables = ['prec', 'tavg']
    fixed_length = 4

    # ds['climate'] = xr.concat([ds[var] for var in vars], dim='band')
    ds['climate'] = xr.concat([ds[variables[0]], ds[variables[1]]], dim='band')
    var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
    ds = ds.assign_coords(band=var_names_U4)
    # ds['climate'] = ds['climate'].assign_coords(band=[s[:fixed_length].ljust(fixed_length) for s in vars])

    ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))

    # ds['climate'] = xr.concat([ds[var] for var in vars], dim='band')
    # keep_vars = ['climate']
    # ds = ds.drop_vars([var for var in ds.data_vars if var not in keep_vars])
    # print(" - add band coordinates")
    # band_var_names = ['prec','tavg']
    # var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in band_var_names]
    # ds = ds.assign_coords(band=var_names_U4)

    # var1='prec'; var2='tavg'
    # ds['climate'] = xr.concat([ds[var1], ds[var2]],
    #                            dim='band')
    # cleanup non-climate vars
    # keep_vars = ['climate']
    # all_vars = list(ds.data_vars)
    # remove_vars = [var for var in all_vars if var not in keep_vars]
    # ds = ds.drop_vars(remove_vars)

    # # add band coordinates
    # print(" - add band coordinates")
    # band_var_names = ['prec','tavg']
    # fixed_length = 4
    # var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in band_var_names]
    # ds = ds.assign_coords(band=var_names_U4)

    # --- clean up types
    print(" - clean up types")
    # month to int type
    ds['month'] = xr.Variable(dims=('month',),
                           data=list(range(1, 12 + 1)))
                           # data=list(range(1, ds.month.shape[0] + 1)))
                           # attrs={'dtype': 'int32'})
    ds["month"] = ds["month"].astype("int32")
    ds["climate"] = ds["climate"].astype("float32")
    ds["band"] = ds["band"].astype("str")
    ds.attrs.clear()


    # # --- force to be like their data
    # print (" - force to be like their data[??]")
    # ds = ds.where(ds.month<=12, drop=True)
    return ds

# sys.exit()

def open_data_srcs(data_srcs):
    print("OPENING ", data_srcs)
    if (len(data_srcs) == 1):
        ds = xr.open_dataset(data_srcs[0])
    else:
        datasets = []
        for f in data_srcs:
            datasets.append(xr.open_dataset(f))
        ds = xr.concat(datasets, dim='time')
    return ds

def rename_dimensions(ds, method, model):
    new_dims = dimensionNames.get_dimension_name(method, model)
    ds = ds.rename(new_dims)
    return ds

def drop_extra_dimensions(ds):
    print(ds)
    vars_to_keep = [time, 'y', 'x', 'prec', 'tavg']
    vars_to_drop = [var for var in ds.variables if var not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)
    return ds

def handle_time_dimension(ds):
    print("Creating spatial and monthly average")
    print("   - if silent failure, run on interactive node")

    # --- smaller time slice for testing
    # print("REMOVE 1999-2001 TIME SLICE")
    # ds = ds.sel(time=slice("1999","2001"))

    spatial_avg_precip = ds['prec'].mean(dim=['y', 'x'])
    spatial_avg_temp = ds['tavg'].mean(dim=['y', 'x'])
    monthly_avg_precip = spatial_avg_precip.resample(time='MS').mean(dim='time')
    monthly_avg_temp = spatial_avg_temp.resample(time='MS').mean(dim='time')

    ds = xr.Dataset({'prec': monthly_avg_precip,
                     'tavg': monthly_avg_temp
    #                   # 't_min': monthly_min_temp,
    #                   # 't_max': monthly_max_temp
                  })

    return ds

def write_obs_to_zarr(ds, ob):
    print("Writing ob to zarr format")
    save_path = f'data/obs256/' + ob + '/' + time_slice_str
    save_f = save_path + '/data.zarr'
    print("Write ob to Zarr file", save_f)
    ds.to_zarr(save_f, consolidated=True)


def write_to_zarr(ds, output_path, method, model, future=False, rcp=None):
    print("Writing to zarr format")
    if (future):
        save_path = output_path + method + '/' + model + '/' + rcp
    else:
        save_path = output_path + method + '/' + model + '/' + time_slice_str
    # print("sp=",save_path)
    # sys.exit()
    # prec_save_path = save_path + '/prec'
    # tavg_save_path = save_path + '/tavg'

    # print("  WARNING: NEED TO ADD LARGER TIME FRAME")
    # --- this block from zarr charts
    # # write precip
    # z_prec = zarr.open(prec_save_path, mode='w', shape=len(ds.prec),
    #               compressor=None, dtype=np.float32)
    # z_prec[:] = ds.prec.data

    # # write temp
    # z_temp = zarr.open(tavg_save_path, mode='w', shape=len(ds.tavg),
    #               compressor=None, dtype=np.float32)
    # z_temp[:] = ds.tavg.data

    # # write attributes
    # z_prec.attrs['TIME START'] = '1980'
    # z_temp.attrs['TIME START'] = '1970'

    save_f = save_path + '/data.zarr'
    # save_f = save_path + '/tavg-prec-month.zarr'
    print("Write to Zarr file", save_f)
    # sys.exit()
    # write the pyramid to zarr, defaults to zarr_version 2
    # consolidated=True, metadata files will have the information expected by site
    # dt = convert_to_zarr_format(ds)
    ds.to_zarr(save_f, consolidated=True)


def convert_to_zarr_format(ds):
    # --- create the pyramid
    print("Convert to Zarr Format")
    print("-- Create Pyramid")
    # EPSG:4326 fixes error:
    #    MissingCRS: CRS not found. Please set the CRS with 'rio.write_crs()'
    print("ds3 =", ds)


    ds.rio.write_crs('EPSG:4326', inplace=True)

    fillValue = 3.4028234663852886e38 # end result red everywhere, inf in np array??
    # fillValue = 9.969209968386869e36 # end result is? NORMAL??
    ds = ds.fillna(fillValue)
    # print(ds)
    # sys.exit()


    # print("debugging 1: pyramid_coarsen instead of reproject")
    # dt = ndp.pyramid_coarsen(ds,
    #                          # factors=[8,4,2,1],
    #                          factors=[2,1],
    #                          dims=['x','y'],
    #                          # boundary='trim',
    #                          )
    # print("ds4 =", ds)

    # HAVE BEEN USING THIS ONE!
#     dt = ndp.pyramid_reproject(ds,
#                                levels=2,
#                                # pixels_per_tile=1024,   # 256 mb
#                                # pixels_per_tile=2048, # 1 gb
#                                # pixels_per_tile=4096, # 4 gb, 13 minutes?
# # Data variables:
# # climate (band, y, x) float32 4GB dask.array<chunksize=(16, 4096, 4096), meta=np.ndarray>
#                                # pixels_per_tile=8192, # 16 gb?



#                                extra_dim='band')
    dt = ndp.pyramid_reproject(ds,
                               levels=2,
                               # pixels_per_tile=PIXELS_PER_TILE,
                               extra_dim='band')
                               # levels=1, # THIS DIDN'D DO MUCH AT ALL
    # dt = dt.rename({'x':'lon', 'y':'lat'})
    print("_______")
    print(dt)
    # print(dt.attrs)
    # print("- - -")
    # print(ds)
    # sys.exit()

    # print("---- Renaming ----")

    # # ds_out = xr.Dataset(
    # #     {
    # #         "lat": ds.lat,
    # #         "lon": ds.lon
    # #     }
    # # )

    # # print(ds_out)


    # print("==== Regridding ====")

    # ds = ds.rename({'x':'lon', 'y':'lat'})
    # print(ds)
    # dt = ndp.pyramid_regrid(ds,
    #                         levels=1,
    #                         pixels_per_tile=256, # 360 * 8, 22.5*128
    #                         regridder_apply_kws={'skipna':True}
    #                         # pixels_per_tile=2944 # 23*128
    #                         # pixels_per_tile=256 # 23*128
    #                         )
    # print(dt.attrs)
    # print("fine")
    # sys.exit()



    # print("debugging 2")
    # dt = ndp.utils.add_metadata_and_zarr_encoding(dt,
    #                                               levels=2,
    #                                               # levels=LEVELS,
    #                                               pixels_per_tile=PIXELS_PER_TILE)
    print("Done Creating Pyramid")
    return dt

if __name__ == "__main__":
    main()
    # comparisons()
