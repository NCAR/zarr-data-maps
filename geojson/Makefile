file=/path/to/region_file.nc
iter=$(shell seq 1 9)

all: region

region: clean
	@for i in $(iter); do \
		python create_geojson.py \
		$(file) \
		--region $$i ; \
	done

clean:
	rm -f *~ foo.geojson
