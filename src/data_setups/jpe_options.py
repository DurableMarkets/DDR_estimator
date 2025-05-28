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
        params_update, options_update, specification, pricing_options, scrap_options, kwargs = get_setup_1(setup_name)
    elif setup_name == 'setup_jpe':
        params_update, options_update, specification, pricing_options, scrap_options, kwargs = get_setup_jpe(setup_name)

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

    return params_update, options_update, specification, pricing_options, scrap_options, out_folders, kwargs

def get_setup_1(setup_name): 
    num_consumers = 8
    num_car_types = 4
    max_age_car = 25 # Unsure about this one. Should probably be 25 but the JPE paper uses 22.

    params_update = {
    "disc_fac": 0.95,
    #"pnew": prices["new_car_prices"],
    "transcost": 0.0,
    "ptranscost": 0.0,
    # "mum2sigma": 0.5,
    #"pscrap": np.array([6.1989, 5.2565, 9.3461, 8.7610]),
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

    # This specification is quite sparse compared to the one in the paper. 
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
                'out_results': lambda setup_name: f'./output/replication/{setup_name}/results/',
                'in_comparison_results': './analysis/data/model_inputs/small_model_scrap_and_price_from_eqb/',
                },
    'max_age_car': max_age_car,
    }

    pricing_options = {
        'how': 'model_moments',
        'new_prices': np.array([174.9022438 , 144.55127776, 299.45192115, 253.39713226]),
        'scrap_prices': np.array([6.1989, 5.2565, 9.3461, 8.7610]),
        'data_source_path': './analysis/data/model_inputs/small_model_scrap_and_price_from_eqb/used_car_prices_model.csv',
        'out_path': f'./analysis/data/{setup_name}/processed_data/',
    }

    scrap_options = {
        'how': 'model_moments',
        'data_source_path': './analysis/data/model_inputs/small_model_scrap_and_price_from_eqb/scrap_probabilities_model.csv',
        'out_path': f'./analysis/data/{setup_name}/processed_data/',
    }

    return params_update, options_update, specification, pricing_options, scrap_options, kwargs


def get_setup_jpe(setup_name): 
    num_consumers = 8
    num_car_types = 4
    max_age_car = 25 # Unsure about this one. Should probably be 25 but the JPE paper uses 22.

    params_update = {
    "disc_fac": 0.95,
    #"pnew": prices["new_car_prices"],
    "transcost": 0.0,
    "ptranscost": 0.0,
    # "mum2sigma": 0.5,
    #"pscrap": np.array([6.1989, 5.2565, 9.3461, 8.7610]),
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

    # This specification is quite sparse compared to the one in the paper. 
    specification = {
    "mum": (num_consumers, 1),
    "buying": (1, 1),
    #"buying": None,
    "scrap_correction": (1, 1),
    "u_0": (num_consumers, num_car_types),
    "u_a": (num_consumers, num_car_types),
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
                'out_results': lambda setup_name: f'./output/replication/{setup_name}/results/',
                'in_comparison_results': './analysis/data/model_inputs/large_model_scrap_and_price_from_eqb/',
                },
    'max_age_car': max_age_car,
    }

    pricing_options = {
        'how': 'model_moments',
        'new_prices': np.array([174.9022438 , 144.55127776, 299.45192115, 253.39713226]),
        'scrap_prices': np.array([6.1989, 5.2565, 9.3461, 8.7610]),
        'data_source_path': './analysis/data/model_inputs/large_model_scrap_and_price_from_eqb/used_car_prices_model.csv',
        'out_path': f'./analysis/data/{setup_name}/processed_data/',
    }

    scrap_options = {
        'how': 'model_moments',
        'data_source_path': './analysis/data/model_inputs/large_model_scrap_and_price_from_eqb/scrap_probabilities_model.csv',
        'out_path': f'./analysis/data/{setup_name}/processed_data/',
    }

    return params_update, options_update, specification, pricing_options, scrap_options, kwargs

