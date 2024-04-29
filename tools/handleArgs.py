import sys

# argument check
def get_arguments(downscaling_methods, climate_models):
    if len(sys.argv) < 2:
        print("Usage: python script.py <dir_path> [<file1> <file2> ...]")
        sys.exit(0)
    data_path = sys.argv[1]
    data_files = sys.argv[2:]
    fullpaths = combine_path_and_files(data_path, data_files)
    method, model = get_method_and_model(fullpaths[0], downscaling_methods, climate_models)
    return fullpaths, method, model

def get_method_and_model(f, downscaling_methods, climate_models):
    found = False
    method = None
    model = None
    for m in downscaling_methods:
        if (m in f):
            if found:
                print("Error: multiple methods found in path")
            found = True
            method = m
    found = False
    for m in climate_models:
        if (m in f):
            if found:
                print("Error: multiple models found in path")
            found = True
            model = m

    if (method == None) or (model == None):
        print("Error: method =", method, ", model =", model)
    return method.lower(), model.lower()

def combine_path_and_files(data_path, data_files):
    combined_paths = []
    for f in data_files:
        combined_paths.append(data_path + '/' + f)
    return combined_paths

# comparison argument check
def get_comparison_arguments(downscaling_methods, climate_models, time_slice_strs):
    if len(sys.argv) < 2:
        print("Usage: python script.py <dir_path>")
        sys.exit(0)
    data_path = sys.argv[1]
    zarr_input_file = sys.argv[2]
    print(data_path)
    print(time_slice_strs)

    comparison_paths = []

    for method in downscaling_methods:
        for model in climate_models:
            for time_slice in time_slice_strs:
                f = data_path + '/' + method + '/' + model + '/' + time_slice + '/' + zarr_input_file
                comparison_paths.append(f)
    # data_path = sys.argv[1]
    # data_files = sys.argv[2:]
    # fullpaths = combine_path_and_files(data_path, data_files)
    # method, model = get_method_and_model(fullpaths[0], downscaling_methods, climate_models)
    return comparison_paths
    # return fullpaths, method, model
