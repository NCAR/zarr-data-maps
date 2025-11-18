# Regional Geojsons
The script creates a geojson border file of a region.
The script selects the mask variable (default is 'regions') and creates a
geojson file that borders where the mask is equal to the passed `--region` value.

## Run
```
$ create_geojson.py path/to/region_foo.nc --region 4
Creating region Pacific Northwest
Wrote LinearRing to pacific_northwest.geojson
```

## Regions
| Region            | Filename              |
|-------------------|-----------------------|
| North Atlantic    | north\_atlantic.nc    |
| Mid Atlantic      | mid\_atlantic.nc      |
| Gulf Coast        | gulf\_coast.nc        |
| Pacific Northwest | pacific\_northwest.nc |
| Pacific Southwest | pacific\_southwest.nc |
| Northern Plains   | northern\_plains.nc   |
| Mountain West     | mountain\_west.nc     |
| Great Lakes       | great\_lakes.nc       |
| Desert Southwest  | desert\_southwest.nc  |
