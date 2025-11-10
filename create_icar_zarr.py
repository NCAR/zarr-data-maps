import argparse
import os
import pandas as pd
import ndpyramid as ndp
import numpy as np
import rioxarray
import sys
import xarray as xr
# import xesmf as xe
import zarr
from tools import dimensionNames, handleArgs

print("ndpyramid Version = ", ndp.__version__)

LEVELS = 4
# LEVELS = 1
PIXELS_PER_TILE = 512 # this one too high
PIXELS_PER_TILE = 256


# Define the global grid (covering the entire world)
global_lon = np.linspace(-180, 180, int(360/0.125 + 1))  # 0.125-degree resolution
global_lat = np.linspace(-90, 90, int(180/0.125 + 1))    # 0.125-degree resolution

# world_grid = xr.Dataset({
#     'x': (['x'], global_lat),
#     'y': (['y'], global_lon)
# })

world_grid = xr.Dataset({
    'lat': (['lat'], global_lat),
    'lon': (['lon'], global_lon)
})
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
    'ICARwest',
    'GARD_r2',
    'GARD_r3',
    'LOCA_8th',
    'MACA',
    'NASA-NEX',
    ]
    # 'GARDwest',
climate_models = [
    'ACCESS1-3',
    'CanESM2',
    'CCSM4',
    'MIROC5',
    'MRI-CGCM3',
    'NorESM1-M',
#    # 'modelmean',
#    # ADD MRI-CGCM3, NorESM1 but are they there??
    ]
# climate_models = ['MRI-CGCM3']
observation_datasets = [
    'CONUS404',
    'GMET',
    'gridMET', # new
    'Livneh',
    'Maurer', # doesn't have new annual and snow variables
    'nClimGrid', # new
    'NLDAS',
    'oldLivenh',
    'PRISM',]

debug=False
if (debug):
    downscaling_methods = ['ICAR']
    climate_models = ['ACCESS1-3']
    observation_datasets = ['CONUS404']


# set these somewhere else?
VERSION = 2
time = 'time'
# time_slices=[slice("1980","2010"), slice("2070","2100")]
# time_slice_strs=['1980_2010', '2070_2100']
# time_slice=slice("1980","2010")
# time_slice_str='1980_2010'
future_time_slice=slice("2070","2100")
future_time_slice_str='2070_2100'

# testing new
time_slice_str='1980_2010'

time_slice_str='1981_2004'

PAST=0
FUTURE=1
CLIMATE_SIGNAL=2


class Dataset:
    def __init__(self, method=None, model=None, past=None, future=None, metric=None, rcp='', obs=None):
        if (past != None) and not os.path.exists(past):
            print("ERROR: past path does not exist:", past)
            sys.exit()
        if (future != None) and not os.path.exists(future):
            print("ERROR: future path does not exist:", future)
            sys.exit()
        if (obs != None) and not os.path.exists(obs):
            print("ERROR: obs path does not exist:", obs)
            sys.exit()
        if (metric != None) and not os.path.exists(metric):
            print("ERROR: metric path does not exist:", metric)
            sys.exit()
        self.past_path = past
        self.future_path = future
        self.metric_path = metric
        self.method = method
        self.model = model
        self.rcp = rcp
        self.obs = obs
    def print(self):
        print("Dataset:")
        print("  past_path =", self.past_path)
        print("  future_path =", self.future_path)
        print("  metric_path =", self.metric_path)
        print("  method =", self.method)
        print("  model =", self.model)
        print("  obs =", self.obs)


class Options:
    def __init__(self, input_path, input_obs_path, input_obs_file,
                 past_path=None, future_path=None,
                 metric_score_path=None, climate_signal_path=None,
                 obs_path=None):
        self.input_path = input_path
        self.input_obs_path = input_obs_path
        self.input_obs_file = input_obs_path+'/'+input_obs_file
        self.write_past = False
        self.past_path = None
        self.write_future = False
        self.future_path = None
        self.write_metric_score = False
        self.metric_score_path = None
        self.write_climate_signal = False
        self.climate_signal_path = None
        self.write_obs = False
        self.obs_path = None
        if past_path != None:
            self.write_past = True
            self.past_path = self.check_path(past_path)
        if future_path != None:
            self.write_future = True
            self.future_path = self.check_path(future_path)
        if metric_score_path != None:
            self.write_metric_score = True
            self.metric_score_path = self.check_path(metric_score_path)
        if obs_path != None:
            self.write_obs = True
            self.obs_path = self.check_path(obs_path)
        if climate_signal_path != None:
            self.write_climate_signal = True
            self.climate_signal_path = self.check_path(climate_signal_path)
    def check_path(self, path, trailing_slash=True):
        if isinstance(path, list):
            path = path[0]
        if trailing_slash and path[-1] != '/':
            path += '/'
        return path
    def print(self):
        print("Options:")
        print("  input path =", self.input_path)
        print("  input obs path =", self.input_obs_path)
        print("  input obs file =", self.input_obs_file)
        print("  past write =", self.write_past,
              ", path =", self.past_path)
        print("  future write =", self.write_future,
              ", path =", self.future_path)
        print("  metric score write =", self.write_metric_score,
              ", path =", self.metric_score_path)
        print("  climate signal write =", self.write_climate_signal,
              ", path =", self.climate_signal_path)
        print("  obs write =", self.write_obs,
              ", path =", self.obs_path)


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


def writeDatasetToZarr(output_path, dataset,
                       write_past=False, write_future=False,
                       write_climate_signal=False,
                       write_metric_score=False,
                       write_obs=False):
    method = dataset.method
    model = dataset.model

    print("Opening file(s):")
    if (write_climate_signal):
        print("past:", dataset.past_path)
        ds_past = xr.open_dataset(dataset.past_path)
        print("past:", dataset.future_path)
        ds_future = xr.open_dataset(dataset.future_path)
        ds = ds_future - ds_past
    elif (write_metric_score):
        print("past:", dataset.past_path)
        ds_past = xr.open_dataset(dataset.past_path)
        print("obs:", dataset.obs)
        ds_obs = xr.open_dataset(dataset.obs)
        ds = ds_obs
        for var in ds_obs.data_vars:
            ds[var] = abs(ds_past[var] - ds_obs[var])
    elif (write_past):
        print('past:', dataset.past_path)
        ds = xr.open_dataset(dataset.past_path)
    elif (write_future):
        print('future:', dataset.future_path)
        ds = xr.open_dataset(dataset.future_path)
    elif (write_obs):
        print('obs:', dataset.obs)
        ds = xr.open_dataset(dataset.obs)


    # variables for zarr creation, value has to be four characters
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
            'ann_t':'annt',
            'ann_p':'annp',
            'ann_snow':'anns',
            'freezethaw':'fzth',
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
    dz = convert_to_zarr_format(ds) # already in single precision


    # setup write_path
    if method != None:
        method_s = method.lower().replace('-','_')
    if model != None:
        model_s = model.lower().replace('-','_')
    if (write_climate_signal):
        write_path = output_path + method_s + '/' + model_s + '/' + dataset.rcp
    if (write_metric_score):
        write_path = output_path + method_s + '/' + model_s + '/' + dataset.rcp
    elif (write_future):
        write_path = output_path + method_s + '/' + model_s + '/' + \
            future_time_slice_str + '/' + dataset.rcp
    elif (write_past):
        write_path = output_path + method_s + '/' + model_s + '/' + \
            time_slice_str
    elif (write_obs):
        # write_obs_to_zarr(ds, ob.lower().replace('-','_'))
        ob_filename = os.path.basename(dataset.obs).split(".ds")[0]
        write_path = output_path + \
            ob_filename.lower().replace('-','_') + '/' + \
            time_slice_str
        # print("Write ob to Zarr file", write_path)

    write_to_zarr(dz, write_path)
    print("small fin", 'output_path=',output_path)
    sys.exit()



def handlePastFutureArgs(input_path, time_period):
    datasets = []

    if time_period == PAST:
        rcps = ['.hist.1981-2004']
    elif time_period == FUTURE:
        rcps = ['.rcp45', '.rcp85', '.rcp45.2076-2099', '.rcp85.2076-2099']

    for cm in climate_models:
        for dm in downscaling_methods:
            for rcp in rcps:
                filepath = input_path+'/'+cm+'.'+dm+rcp+'.ds.conus.metric.maps.nc'
                if cm == 'modelmean':
                    filepath = input_path+'/'+dm+rcp+'.'+cm+'.ds.conus.metrics.nc'
                # print('f=',filepath)
                if (time_period == PAST and os.path.exists(filepath)):
                    datasets.append(Dataset(dm, cm, past=filepath, rcp=rcp[1:]))
                elif (time_period == FUTURE and os.path.exists(filepath)):
                    datasets.append(Dataset(dm, cm, future=filepath,
                                            rcp=rcp[1:]))
                else:
                    print("DIDN'T ADD f=", filepath)
    return datasets

def handleMetricScoreArgs(input_path, metric_path):
    datasets = []

    for cm in climate_models:
        for dm in downscaling_methods:
            metric_path = input_path+'/'+cm+'.'+dm+'.ds.conus.metrics.nc'
            if os.path.exists(metric_path):
                # datasets.append(Dataset(dm, cm, past_path, obs=obs_path))
                datasets.append(Dataset(dm, cm, metric=metric_path))
            else:
                print("metric path doesn't exist:", metric_path)
    return datasets

def handleClimateSignalArgs(input_path):
    rcps = ['rcp45', 'rcp85']
    datasets = []

    for cm in climate_models:
        for dm in downscaling_methods:
            past_path = input_path+'/'+cm+'.'+dm+'.ds.conus.metric.maps.nc'
            for rcp in rcps:
                future_path = input_path+'/'+cm+'.'+dm+'.'+rcp+'.ds.conus.metric.maps.nc'
                # if ('GARD' in dm):
                #     print(cm,"and",dm)
                #     print("past_path :", past_path)
                #     print("future_path :", future_path)

                if os.path.exists(past_path) and os.path.exists(future_path):
                    # if ('GARD' in dm):
                    #     print('exists for', rcp)
                    datasets.append(Dataset(dm, cm, past_path, future_path, rcp=rcp))
    # print('fin')
    # sys.exit()
    return datasets



def handleObsArgs(input_obs_path):
    datasets = []
    for ob in observation_datasets:
        obs_path = input_obs_path+'/'+ob+'.ds.conus.metric.maps.nc'
        if os.path.exists(obs_path):
                datasets.append(Dataset(obs=obs_path))
    return datasets

# def writeObsToZarr(dataset, output_obs_path):
#     # print("--- Starting Zarr Data Maps Setup ---")
#     # library_check()
#     # paths = handleObsArgs(input_obs_path) OLD  ONE NEED

#     dataset.print()
#     # sys.exit()

#     ob = dataset.obs
#     ds = xr.open_dataset(ob)

#     # for a in ds.variables.keys():
#     #     print(a)
#     # sys.exit()

#     # rename dimensions
#     # 'lat_y': y_name,
#     # 'lon_x': x_name,
#     new_dims = {#'time': 'time',
#                 'lat': 'y',
#                 'lon': 'x',
#                 'n34pr':'n34p',
#                 'ttrend':'ttre',
#                 'ptrend':'ptre',
#                 'pr90':'pr90',
#                 'pr99':'pr99',
#                 't90':'t90_',
#                 't99':'t99_',
#                 'djf_t':'djft',
#                 'djf_p':'djfp',
#                 'mam_t':'mamt',
#                 'mam_p':'mamp',
#                 'jja_t':'jjat',
#                 'jja_p':'jjap',
#                 'son_t':'sont',
#                 'son_p':'sonp',
#     }
#     ds = ds.rename(new_dims)
#     variables = list(ds.variables.keys())
#     variables = [var for var in variables if var not in ['x', 'y']]
#     # print(variables)
#     # sys.exit()
#     # don't need to average and handle time dimension
#     # - [ ] do i need to handle time?
#     # don't need to drop extra dimensions
#     # add climate dimension
#     # add climate (aka variable) dimension
#     print(" - add climate (aka variable) dimension")
#     # variables = ['prec', 'tavg']
#     fixed_length = 4
#     concatenated_vars = []
#     for var_name in variables:
#         concatenated_vars.append(ds[var_name])
#     ds['climate'] = xr.concat(concatenated_vars, dim='band')
#     var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
#     ds = ds.assign_coords(band=var_names_U4)
#     ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
#     # --- clean up types
#     print(" - clean up types")
#     # month to int type
#     ds['month'] = xr.Variable(dims=('month',),
#                               data=list(range(1, 12 + 1)))
#                            # data=list(range(1, ds.month.shape[0] + 1)))
#                            # attrs={'dtype': 'int32'})
#     ds["month"] = ds["month"].astype("int32")
#     ds["climate"] = ds["climate"].astype("float32")
#     ds["band"] = ds["band"].astype("str")
#     ds.attrs.clear()
#     ds = convert_to_zarr_format(ds) # already in single precision

#     # write_obs_to_zarr(ds, ob.lower().replace('-','_'))
#     ob_filename = os.path.basename(ob).split(".ds")[0]
#     print("Writing ob to zarr format")
#     save_path = output_obs_path + \
#                 ob_filename.lower().replace('-','_') + '/' + \
#                 time_slice_str
#     save_f = save_path + '/data.zarr'
#     print("Write ob to Zarr file", save_f)
#     # sys.exit()
#     ds.to_zarr(save_f, consolidated=True)

#     # print(ds)
#     # sys.exit()

#     # print('---fin---')
#     # sys.exit()


# parse command line arguments
def parseCLA():
    # define argument parser
    parser = argparse.ArgumentParser(description="Create Zarr files for ICAR Maps.")
    parser.add_argument("input_path", help="Path to input files")
    parser.add_argument("input_obs_path", help="Path to input observations")
    parser.add_argument("input_obs_file", help="Path to input observations")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose mode")
    group = parser.add_argument_group('Output Data', 'types of output and path to write to')
    group.add_argument("--past", nargs=1, dest="past_path",
                       help="Write past model data to passed path")
    group.add_argument("--future", nargs=1, dest="future_path",
                       help="Write future model data to passed path")
    group.add_argument("--metric-score", nargs=1, dest="metric_score_path",
                       help="Write (model_data - obs_data) dataset")
    group.add_argument("--climate-signal", nargs=1, dest="climate_signal_path",
                       help="Write climate signal data to passed path")
    group.add_argument("--obs", nargs=1, dest="obs_path",
                       help="Write observation dataset to passed path")


    # Parse the arguments
    args = parser.parse_args()

    if (args.past_path == None and
        args.future_path == None and
        args.metric_score_path == None and
        args.climate_signal_path == None and
        args.obs_path == None):
        print("ERROR: past, future, metric-score, or climate-signal options are required")
        print(parser.print_help())
        sys.exit(1)

    options = Options(args.input_path, args.input_obs_path, args.input_obs_file,
                      args.past_path, args.future_path,
                      args.metric_score_path, args.climate_signal_path,
                      args.obs_path)

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
    past_datasets = []
    future_datasets = []
    metric_score_datasets = []
    climate_signal_datasets = []
    obs_datasets = []
    if options.write_past:
        past_datasets = handlePastFutureArgs(options.input_path, PAST)
    if options.write_future:
        future_datasets = handlePastFutureArgs(options.input_path, FUTURE)
    if options.write_metric_score:
        metric_score_datasets = handleMetricScoreArgs(options.input_path,
                                                      options.metric_score_path)
        print("metric score datasets", metric_score_datasets)
        sys.exit()
    if options.write_climate_signal:
        climate_signal_datasets = handleClimateSignalArgs(options.input_path)
    if options.write_obs:
        obs_datasets = handleObsArgs(options.input_obs_path)


    # options.print()
    # sys.exit()

    # --- process datasets to write to zarr
    count = 0
    max_count=999999

    print("msds", metric_score_datasets)

    for dataset in metric_score_datasets:
        count+=1
        options.print()
        dataset.print()
        # question is how to order the datasets now?
        writeDatasetToZarr(options.metric_score_path, dataset,
                           write_metric_score = True)
        if (count > max_count):
            print(max_count, "max count reached")
            break

    for dataset in climate_signal_datasets:
        print("Attempting to write ds to zarr for climate signal")
        count+=1
        writeDatasetToZarr(options.climate_signal_path, dataset,
                           write_climate_signal = True)
        if (count > max_count):
            print(max_count, "max count reached")
            break

    for dataset in future_datasets:
        count+=1
        writeDatasetToZarr(options.future_path, dataset,
                           write_future = True)
        if (count > max_count):
            print(max_count, "max count reached")
            break

    for dataset in past_datasets:
        count+=1
        writeDatasetToZarr(options.past_path, dataset,
                           write_past = True)
        if (count > max_count):
            print(max_count, "max count reached")
            break

    for dataset in obs_datasets:
        count+=1
        writeDatasetToZarr(options.obs_path, dataset,
                           write_obs = True)
        if (count > max_count):
            print(max_count, "max count reached")
            break




    print('---fin---')
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

def write_to_zarr(ds, output_path, zarr_file='data.zarr'):
    print("Writing to zarr format")

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

    save_f = output_path + '/' + zarr_file
    if (os.path.exists(save_f)):
        print("ERROR WRITING ZARR FILE: path exists at", save_f)

    print("Write to Zarr file", save_f)
    # sys.exit()
    # write the pyramid to zarr, defaults to zarr_version 2
    # consolidated=True, metadata files will have the information expected by site
    # dt = convert_to_zarr_format(ds)
    ds.to_zarr(save_f, consolidated=True) #, encoding={"zlib":True})
    print("Done writing to zarr format")


def convert_to_zarr_format(ds):
    # --- create the pyramid
    print("Convert to Zarr Format")
    print("-- Create Pyramid")
    # EPSG:4326 fixes error:
    #    MissingCRS: CRS not found. Please set the CRS with 'rio.write_crs()'
    print("ds3 =", ds)


    # write CRS data in Climate and Forecast (CF) Metadata Convention style
    ds.rio.write_crs('EPSG:4326', inplace=True)

    fillValue = 3.4028234663852886e38 # end result red everywhere, inf in np array??
    # fillValue = 9.969209968386869e36 # end result is? NORMAL??
    ds = ds.fillna(fillValue)

    # return ds



    # print(ds)
    # sys.exit()
    # us_grid = xr.Dataset({
    #     'lat': (['lat'], ds['y'].values),
    #     'lon': (['lon'], ds['x'].values)
    # })
    # regridder = xe.Regridder(us_grid,
    #                          world_grid,
    #                          method='conservative',)
    #                          # extrap_method='nearest_s2d')
    #                          # extrap_method='ESMF_EXTRAPMETHOD_NONE')

    # ds = ds.rename({'x':'lon', 'y':'lat'})
    # ds = regridder(ds)
    # ds = ds.rename({'lon':'x', 'lat':'y'})

    # def sel_coarsen(ds, factor, dims, **kwargs):
    #     return ds.sel(**{dim: slice(None, None, factor) for dim in dims})
    # dt = ndp.pyramid_create(ds,
    #                         # factors=[8,4,2,1],
    #                         factors=[1],
    #                         dims=['x','y'],
    #                         func=sel_coarsen,
    #                          # boundary='trim',
    #                          )


    # # # # print("debugging 1: pyramid_coarsen instead of reproject")
    # dt = ndp.pyramid_coarsen(ds,
    #                          # factors=[8,4,2,1],
    #                          factors=[1],
    #                          dims=['x','y'],
    #                          # boundary='trim',
    #                          )
    # # this is needed to get zlib compression, required by mapping js app
    # dt = ndp.utils.add_metadata_and_zarr_encoding(dt,
    #                                               levels=1,)
    #                                               # # levels=LEVELS,
    #                                               # pixels_per_tile=PIXELS_PER_TILE)


    # print("ds4 =", ds)

    # HAVE BEEN USING THIS ONE!
#     dt = ndp.pyramid_reproject(ds,
#                                levels=1,
#                                # pixels_per_tile=1024,   # 256 mb
#                                # pixels_per_tile=1024,   # 256 mb
#                                pixels_per_tile=2048, # 1 gb
#                                # pixels_per_tile=4096, # 4 gb, 13 minutes?
# # Data variables:
# # climate (band, y, x) float32 4GB dask.array<chunksize=(16, 4096, 4096), meta=np.ndarray>



#                                # pixels_per_tile=8192, # 16 gb?
#                                extra_dim='band')
#---- FOO THIS IS ORIGINAL, WHAT I"VE BEEN USING TIL MID 08.2024

    # Pad the dataset with zeros to prevent unwanted interpolation artifacts
    pad_width = 3  # Adjust padding as needed
    # print("--PADDED--")
    # ds_padded = ds.pad(x=pad_width, y=pad_width, constant_values=0)
   # print(ds_padded)
    pixels = 512
    pixels = 13875 / 4
    pixels = 3469
    pixels = 2880 # for equidistant-cylindrical
    pixels = 1440 # test
    pixels = 720 # test
    pixels = 360 # test, this reaches 2880 for level 4

    dt = ndp.pyramid_reproject(ds,
    # dt = ndp.pyramid_reproject(ds_padded,
                               levels=LEVELS,
                               pixels_per_tile=pixels,
                               projection='equidistant-cylindrical',
                               # pixels_per_tile=PIXELS_PER_TILE,
                               extra_dim='band')
                               # levels=1, # THIS DIDN'D DO MUCH AT ALL
    # print("_______ printing dt _______")
    # print(dt)
    # sys.exit()

#---- 08.2024
    # dt = dt.rename({'x':'lon', 'y':'lat'})
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


   #  print("==== Regridding ====")

   #  # fixes ValueError: dataset must include lon/lat or be CF-compliant
   #  ds = ds.rename({'x':'lon', 'y':'lat'})
   #  # print(ds)
   #  dt = ndp.pyramid_regrid(ds,
   #                          levels=1,
   #                          # regridder_apply_kws={'skipna':True},
   # # extra_dim='band', DOESNT WORK ON PYRAMID REGRID JUST REPROJECT
   #                          # method options
   #                          method='conservative',
   #                          # regridder_apply_kws={
   #                          #     'extrap_method':'ESMF_EXTRAPMETHOD_NONE'},
   #                          # method='nearest_d2s', # best so far
   #                          # method='bilinear',
   #                          # method='patch',
   #                          # pixels_per_tile=256, # 360 * 8, 22.5*128
   #                          # pixels_per_tile=2944 # 23*128
   #                          # pixels_per_tile=256,
   #                          # pixels_per_tile=512,
   #                          # pixels_per_tile=1024, # this one if have to
   #                          pixels_per_tile=2048, # THIS TAKES FOREVER
   #                          )
   #  # going back to what carbonplan's map needs
   #  ds = ds.rename({'lon':'x', 'lat':'y'})
    # print(dt.attrs)
    # print("fine")

    # print("_______")
    # print(dt)
    # sys.exit()



    # print("debugging 2")
    # dt = ndp.utils.add_metadata_and_zarr_encoding(dt,
    #                                               levels=1,
    #                                               # levels=LEVELS,
    #                                               pixels_per_tile=PIXELS_PER_TILE)
    print("Done Creating Pyramid")
    return dt

if __name__ == "__main__":
    main()
    # comparisons()
