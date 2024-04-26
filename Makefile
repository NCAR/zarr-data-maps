lc=python3

all: host

host:
	$(lc) host_server.py

build: create_zarr_data

create_zarr_data:
	$(lc) create_icar_zarr.py

create_zarr_charts:
	$(lc) create_icar_charts.py

clean:
	rm -f *~
cleandata:
	rm -rf downscaling
