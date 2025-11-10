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
`$ mamba install ipykernel`  # testing mamba install ipytree ipykernel to see if order helps
`$ python -m ipykernel install --user --name=zarr`

### Create Data
Setup paths in `create_icar_zarr_.py`. Will become command line arguments in the future.
`$ python create_icar_zarr.py`

#### Notes on Data Creation
* Data variables must be dimensions of a four dimensional `climate` variable.
The `climate` variables dimensions will be `x`, `y`, `band`, `month`.
The `month` requirement can be updated in the Maps site's `components/parameter-controls.js` file.
* Note variable names must be of type `U4`, Python strings of length 4.



## Existing Zarr Data Datasets
Current Zarr dataset that are being used by [hydro climate evaluation sites](https://github.com/NCAR/hydro-climate-evaluation) can be found at [hydro.rap.ucar.edu/hydro-climate-eval/data](https://hydro.rap.ucar.edu/hydro-climate-eval/data/).

## Hosting
The files [host_erver.py](https://github.com/scrasmussen/zarr-data-maps/blob/main/host_server.py) or hydro-climate evaluation's [server.js](https://github.com/NCAR/hydro-climate-evaluation/blob/main/server.js) offer examples of how to make the Zarr data available to be read by a website.

### Host locally
#### Prepare Data
Either [create data](#create-data) or copy it over.
To copy the data edit the [Makefile](Makefile) and run `make scp untar` to copy the data from Derecho and untar it.
#### Host Site
Once data is created the user can host the data locally for local development.
Running `make host` or the following command will start a local server.
```
$ python host_server.py
```
