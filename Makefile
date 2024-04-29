lc=python3
file=create_icar_zarr.py
file=create_icar_zarr_charts.py


icar_path=/glade/campaign/ral/hap/trude/conus_icar/qm_data

# 1980-2010
icar_noresm_files=noresm_hist_exl_conv.nc
icar_cesm_files=cesm_hist_exl_conv.nc
icar_gfdl_files=gfdl_hist_exl_conv.nc
icar_miroc5_files=miroc5_hist_exl_conv.nc

# 2070-2100
icar_noresm_files=noresm_rcp45_exl_conv.nc
icar_cesm_files=cesm_rcp45_exl_conv.nc
icar_gfdl_files=gfdl_rcp45_exl_conv.nc
icar_miroc5_files=miroc5_rcp45_exl_conv.nc

icar_zarr_path=data/map

all: build_comparison


build_comparison:
	python3 create_icar_zarr.py ${icar_zarr_path} tavg-prec-month.zarr

build_icar:
	python3 create_icar_zarr.py ${icar_path} ${icar_noresm_files}
	python3 create_icar_zarr.py ${icar_path} ${icar_cesm_files}
	python3 create_icar_zarr.py ${icar_path} ${icar_gfdl_files}
	python3 create_icar_zarr.py ${icar_path} ${icar_miroc5_files}

build_icar_charts:
	# python3 create_icar_zarr_charts.py ${icar_path} ${icar_noresm_files}
	# python3 create_icar_zarr_charts.py ${icar_path} ${icar_cesm_files}
	# python3 create_icar_zarr_charts.py ${icar_path} ${icar_gfdl_files}
	# python3 create_icar_zarr_charts.py ${icar_path} ${icar_miroc5_files}


tar:
	cd data/map; tar -czvf icar.tar.gz icar


clean:
	rm -f *~
cleandata:
	rm -rf icar_zarr
	# $(lc) $(file)
