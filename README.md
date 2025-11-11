# ICAR Zarr Data
This repository is for creating and hosting Zarr output for the [hydro-climate evaluation](https://github.com/NCAR/hydro-climate-evaluation) set of websites.
To use existing data, see [Host ICAR Zarr Data](#host-icar-zarr-data) to copy existing files and start a local server or host a server from Derecho.
To create Zarra dataset see [Create ICAR Zarr Data](#create-icar-zarr-data) for instructions on generating Zarr data from existing NetCDF output.

## Create ICAR Zarr Data
Create Zarr data files in a format that can be accessed by [carbonplan/maps](https://github.com/carbonplan/maps) websites.
This is setup for generation of data that can be access locally by [hydro-climate evaluation](https://github.com/NCAR/hydro-climate-evaluation).

### Requirements
Setup and activate conda environment.
`$ conda install --file requirements.txt`

#### Setting up Jupyter environment
Not required but could help with development if user wants to use a Jupyter notebook.
```
# testing mamba install ipytree ipykernel to see if order helps
$ mamba install ipykernel
$ python -m ipykernel install --user --name=zarr
```

### Create Data
```bash
$ python create_global_zarr.py --help
usage: create_global_zarr.py [-h] [-v] [--climate-signal CLIMATE_SIGNAL_PATH] [--maps OUTPUT_MAPS_PATH] [--metric-score METRIC_SCORE_PATH] [--obs OUTPUT_OBS_PATH]
                             [--time-series TIMESERIES_PATH] [--write-yaml] [--test]
                             input_maps_path input_obs_path input_metrics_path

Create Zarr files for ICAR Maps.

positional arguments:
  input_maps_path       Path to input map files
  input_obs_path        Path to input observations
  input_metrics_path    Path to input metrics

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose mode
  --write-yaml          Write Yaml Files
  --test                Test inputs

Output Data:
  types of output and path to write to

  --climate-signal CLIMATE_SIGNAL_PATH
                        Write climate signal data to passed path
  --maps OUTPUT_MAPS_PATH
                        Write modeled data to passed path
  --metric-score METRIC_SCORE_PATH
                        Write (model_data - obs_data) dataset
  --obs OUTPUT_OBS_PATH
                        Write observation dataset to passed path
  --time-series TIMESERIES_PATH
                        Write time series dataset to passed path
```

The script requires paths to input NetCDF files of the maps, observations, and metrics.
The user can choose what to output by choosing one of the optional options
`--climate-signal, --maps, --metric-score, --obs` or `--time-series`.
The following is an example of a command to write the Zarr maps:
```bash
$ python3 create_global_zarr.py \
    path/to/maps \
    path/to/obs \
    path/to/metrics \
    --maps output/maps
```

#### Notes on Data Creation
* Data variables must be dimensions of a four dimensional `climate` variable.
The `climate` variables dimensions will be `x`, `y`, `band`, `month`.
The `month` requirement can be updated in the Maps site's `components/parameter-controls.js` file.
* Note variable names must be of type `U4`, Python strings of length 4. The `carbonplan/maps` requires this formatting choice.


## Existing Zarr Data Datasets
Current Zarr dataset that are being used by [hydro climate evaluation sites](https://github.com/NCAR/hydro-climate-evaluation) can be found at [hydro.rap.ucar.edu/hydro-climate-eval/data](https://hydro.rap.ucar.edu/hydro-climate-eval/data/).

## Hosting
The files [host_server.py](https://github.com/scrasmussen/zarr-data-maps/blob/main/host_server.py) or hydro-climate evaluation's [server.js](https://github.com/NCAR/hydro-climate-evaluation/blob/main/server.js) offer examples of how to make the Zarr data available to be read by a website.

### Host locally
#### Prepare Data
Either [create data](#create-data) or copy and untar it over from the [hydro.rap.ucar.edu/hydro-climate-eval/data](https://hydro.rap.ucar.edu/hydro-climate-eval/data/).

#### Host Site
Once data is created the user can host the data locally for local development.
Running `make host` or the following command will start a local server.
```
$ python host_server.py
```
