# ICAR Zarr Data
This repository is for creating and hosting Zarr output for the [ICAR Maps](https://github.com/scrasmussen/icar-maps) website.
To use existing data, see [Host ICAR Zarr Data](#host-icar-zarr-data) to copy existing files and start a local server.
To create Zarra dataset see [ICAR Zarr Data](#icar-zarr-data) to copy existing files and start a local server.


## Host ICAR Zarr Data
After choosing one of the following two options to host the data, it will be available at [localhost:4000](http://localhost:4000) and can be accessed from a browser.

### Host on Derecho
1. Login to Derecho and start server locally. Note or double check the login node number accessed with `$ echo $HOST`.
```
$ python host_server.py
```
2. From local computer SSH into the login number hosting the server. The `-L` argument enables port forwarding.
```
ssh -L 4000:localhost:4000 username@derecho[1-7].hpc.ucar.edu
```


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




## Create ICAR Zarr Data
Create Zarr data files in a format that can be accessed by [carbonplan/maps](https://github.com/carbonplan/maps) websites.
This is setup for generation of data that can be access locally by [icar/maps](https://github.com/scrasmussen/icar-maps).

### Requirements
Setup and activate conda environment.
`$ conda install --file requirements.txt`

#### Setting up Jupyter environment
`$ mamba install ipykernel`  # testing mamba install ipytree ipykernel to see if order helps
`$ python -m ipykernel install --user --name=zarr`

### Create Data
Setup paths in `create_icar_zarr_.py`. Will become command line arguments in the future.
`$ python3 create_icar_zarr.py`

#### Notes on Data Creation
* Data variables must be dimensions of a four dimensional `climate` variable.
The `climate` variables dimensions will be `x`, `y`, `band`, `month`.
The `month` requirement can be updated in the Maps site's `components/parameter-controls.js` file.
* Note variable names must be of type `U4`, Python strings of length 4.
