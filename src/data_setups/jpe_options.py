import numpy as np
import yaml 
import os
from data_setups.tools import create_folder
def get_model_specs():
    # Load yaml file name 

    with open(os.path.join(os.path.dirname(__file__), "pick_setup.yml"), 'r') as stream:
        options = yaml.safe_load(stream)
    setup_name = options['jpe_setup_name']
    # load sim options 
    if setup_name == 'setup_1':
        params_update, options_update, specification, kwargs = get_setup_1()
    else: 
        raise ValueError(f"Invalid setup name: {setup_name}. Please choose a valid JPE setup name.")

    folders = kwargs['folders']
    out_folders = {}
    for key, value in folders.items():
        if callable(value):
            create_folder(value(setup_name))
            out_folders[key] = value(setup_name)
        elif isinstance(value, str): 
            create_folder(value)
            out_folders[key] = value
        else: 
            raise ValueError(f"Invalid folder value: {value}. Please provide a valid folder path or function.")

    return params_update, options_update, specification, out_folders, kwargs

def get_setup_1(): 
    num_consumers = 8
    num_car_types = 4
    max_age_car = 25 # Unsure about this one. Should probably be 25 but the JPE paper uses 22.

    params_update = {
    "disc_fac": 0.95,
    #"pnew": prices["new_car_prices"],
    "transcost": 0.0,
    "ptranscost": 0.0,
    # "mum2sigma": 0.5,
    "pscrap": np.array([6.1989, 5.2565, 9.3461, 8.7610]),
    "acc_0": np.array([-4.6363, -4.6363, -4.6363, -4.6363]),
    "acc_a": np.array([0.0, 0.0, 0.0, 0.0]),
    "acc_even": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    options_update = {
        "n_consumer_types": num_consumers, # Redundant
        "n_car_types": num_car_types,
        "max_age_of_car_types": [max_age_car],
        #"tw": [0.5, 0.5],
    }

    specification = {
    "mum": (num_consumers, 1),
    "buying": (1, 1),
    #"buying": None,
    "scrap_correction": (1, 1),
    "u_0": (1, num_car_types),
    "u_a": (1, num_car_types),
    "u_a_sq": None,
    "u_a_even": None,
    }
    
    # additional options that control output folders, choice of years to estimate on and the max 
    kwargs = {
    #'indir': './analysis/data/8x4/',
    #'outdir': './analysis/data/8x4_eqb/', # gets added after get_model_specs
    'years': np.arange(1996, 2009).astype(str).tolist(),
    'folders': {'in_data': './analysis/data/8x4/',
                'out_data':lambda setup_name: f'./analysis/data/{setup_name}/processed_data/',
                'out_results': lambda setup_name: f'./output/replication/{setup_name}/results/'},
    'max_age_car': max_age_car,
    }
    return params_update, options_update, specification, kwargs
