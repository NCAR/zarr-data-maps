"""
dimensionNames: returns dictionary for converting downscaling and climate
                model files into ICAR Map website format
"""

time_name = 'time' # changed to month later
y_name = 'y'
x_name = 'x'
prec_name = 'prec'
tavg_name = 'tavg'

dimensions = {
    # Time Dimensions
    # 'time': time_name,
    # grid dimensions
    'lat_y': y_name,
    'lon_x': x_name,
    'lat': y_name,
    'lon': x_name,

    # --- variables ---
    # precipitation
    'precipitation': prec_name,
    'pcp': prec_name,
    # average temperature
    'ta2m': tavg_name,
    't_mean': tavg_name,
}

# ICAR + climate models dimensions
icar_noresm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
icar_cesm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
icar_gfdl_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
icar_miroc5_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}

# GARD + climate models dimensions
gard_noresm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
gard_cesm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
gard_gfdl_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
gard_miroc5_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}

# LOCA + climate models dimensions
loca_noresm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
loca_cesm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
loca_gfdl_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
loca_miroc5_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}

# BCSD + climate models dimensions
bcsd_noresm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
bcsd_cesm_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
bcsd_gfdl_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}
bcsd_miroc5_dims = {
    'time': time_name,
    'lat': y_name,
    'lon': x_name,
    'pcp': prec_name,
    't_mean': tavg_name,
}


def get_dimension_name(method, model):
    dims = None
    print("Method", method, "model", model)
    if method == 'icar':
        if model == 'noresm':
            dims = icar_noresm_dims
        elif model == 'cesm':
            dims = icar_cesm_dims
        elif model == 'gfdl':
            dims = icar_gfdl_dims
        elif model == 'miroc5':
            dims = icar_miroc5_dims

    elif method == 'gard':
        if model == 'noresm':
            dims = gard_noresm_dims
        elif model == 'cesm':
            dims = gard_cesm_dims
        elif model == 'gfdl':
            dims = gard_gfdl_dims
        elif model == 'miroc5':
            dims = gard_miroc5_dims

    elif method == 'loca':
        if model == 'noresm':
            dims = loca_noresm_dims
        elif model == 'cesm':
            dims = loca_cesm_dims
        elif model == 'gfdl':
            dims = loca_gfdl_dims
        elif model == 'miroc5':
            dims = loca_miroc5_dims

    elif method == 'icar':
        if model == 'noresm':
            dims = icar_noresm_dims
        elif model == 'cesm':
            dims = icar_cesm_dims
        elif model == 'gfdl':
            dims = icar_gfdl_dims
        elif model == 'miroc5':
            dims = icar_miroc5_dims


    return dims
