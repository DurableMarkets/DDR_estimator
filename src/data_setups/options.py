import numpy as np
import yaml 
import os
def get_model_specs(output_dir_func):
    # Load yaml file name 

    with open(os.path.join(os.path.dirname(__file__), "pick_setup.yml"), 'r') as stream:
        options = yaml.safe_load(stream)
    setup_name = options['name']
    # load sim options 
    if setup_name == 'setup_1':
        sim_options, mc_options, params_update, options_update, specification = get_setup_1()
    elif setup_name == 'setup_2':
        sim_options, mc_options, params_update, options_update, specification = get_setup_2()
    elif setup_name == 'setup_3':
        sim_options, mc_options, params_update, options_update, specification = get_setup_3()
    else: 
        raise ValueError("Invalid setup name. Please choose a valid setup name.")

    output_dir= output_dir_func(setup_name)
    create_folder(output_dir)
    return sim_options, mc_options, params_update, options_update, specification, output_dir

def sim_option_checks(estimation_size, chunk_size, N_mc, sample_iter):
        assert (
        estimation_size % chunk_size == 0
        ), "estimation_size should be a multiple of chunk_size"
        assert N_mc % chunk_size == 0, "N_mc should be a multiple of chunk_size"
        assert (
            estimation_size <= chunk_size * sample_iter
        ), "estimation_size should be smaller or equal to chunk_size * mc_iter"

def create_folder(folder_name):
    """
    Create a folder if it does not exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

def get_setup_1(): 
    num_consumers = 2
    num_car_types = 2

    chunk_size = 250_000
    mc_iter = 100
    N_mc = 500_000 #5_000_000 
    sample_iter = N_mc * mc_iter // chunk_size
    # Estimation_size controls the sample size used in the estimation
    estimation_size = N_mc  # 1000000
    
    sim_option_checks(estimation_size, chunk_size, N_mc, sample_iter)

    sim_options = {
        "n_agents": chunk_size * sample_iter,  # 226675,
        "n_periods": 1,
        "seed": 500,
        "chunk_size": chunk_size,
        "estimation_size": estimation_size,
        "use_count_data": True,
    }

    mc_options = {
    'Nbars': np.array([estimation_size]),
    'mc_iter': mc_iter, 
    }

    params_update = {
        "p_fuel": [0.0],
        "acc_0": [-100.0],
        "mum": [0.5, 0.5],
        "psych_transcost": [2.0, 2.0],
        'u_0': np.array([[12.0,12.0],[12.0,12.0]]),
    }

    options_update = {
        "n_consumer_types": num_consumers, # Redundant
        "n_car_types": num_car_types,
        "max_age_of_car_types": [25],
        "tw": [0.5, 0.5],
    }

    specification = {
    "mum": (num_consumers, 1),
    "buying": (num_consumers, 1),
    #"buying": None,
    "scrap_correction": (num_consumers, 1),
    "u_0": (num_consumers, num_car_types),
    "u_a": (num_consumers, 1),
    "u_a_sq": None,
    "u_a_even": None,
    }

    return sim_options, mc_options, params_update, options_update, specification

def get_setup_2(): 
    num_consumers = 2
    num_car_types = 2

    chunk_size = 250_000
    mc_iter = 100
    N_mc = 500_000 #5_000_000 
    sample_iter = N_mc * mc_iter // chunk_size
    # Estimation_size controls the sample size used in the estimation
    estimation_size = N_mc  # 1000000
    
    sim_option_checks(estimation_size, chunk_size, N_mc, sample_iter)

    sim_options = {
        "n_agents": chunk_size * sample_iter,  # 226675,
        "n_periods": 1,
        "seed": 500,
        "chunk_size": chunk_size,
        "estimation_size": estimation_size,
        "use_count_data": True,
    }

    mc_options = {
    'Nbars': np.array([estimation_size]),
    'mc_iter': mc_iter, 
    }

    params_update = {
        "p_fuel": [0.0],
        "acc_0": [-100.0],
        "mum": [0.35, 0.5],
        "psych_transcost": [2.0, 2.0],
        'u_0': np.array([[12.0,12.0],[12.0,12.0]]),
    }

    options_update = {
        "n_consumer_types": num_consumers, # Redundant
        "n_car_types": num_car_types,
        "max_age_of_car_types": [25],
        "tw": [0.5, 0.5],
    }

    specification = {
    "mum": (num_consumers, 1),
    "buying": (num_consumers, 1),
    #"buying": None,
    "scrap_correction": (num_consumers, 1),
    "u_0": (num_consumers, num_car_types),
    "u_a": (num_consumers, 1),
    "u_a_sq": None,
    "u_a_even": None,
    }

    return sim_options, mc_options, params_update, options_update, specification



def get_setup_3(): 
    num_consumers = 2
    num_car_types = 2

    chunk_size = 250_000
    mc_iter = 100
    N_mc = 500_000 #5_000_000 
    sample_iter = N_mc * mc_iter // chunk_size
    # Estimation_size controls the sample size used in the estimation
    estimation_size = N_mc  # 1000000
    
    sim_option_checks(estimation_size, chunk_size, N_mc, sample_iter)

    sim_options = {
        "n_agents": chunk_size * sample_iter,  # 226675,
        "n_periods": 1,
        "seed": 500,
        "chunk_size": chunk_size,
        "estimation_size": estimation_size,
        "use_count_data": True,
    }

    mc_options = {
    'Nbars': np.array([estimation_size]),
    'mc_iter': mc_iter, 
    }

    params_update = {
        "p_fuel": [0.0],
        "acc_0": [-100.0],
        "mum": [0.35, 0.5],
        "psych_transcost": [2.0, 2.0],
        'u_0': np.array([[7.0,9.0],[7.0,6.0]]),
    }

    options_update = {
        "n_consumer_types": num_consumers, # Redundant
        "n_car_types": num_car_types,
        "max_age_of_car_types": [25],
        "tw": [0.5, 0.5],
    }

    specification = {
    "mum": (num_consumers, 1),
    "buying": (num_consumers, 1),
    #"buying": None,
    "scrap_correction": (num_consumers, 1),
    "u_0": (num_consumers, num_car_types),
    "u_a": (num_consumers, 1),
    "u_a_sq": None,
    "u_a_even": None,
    }

    return sim_options, mc_options, params_update, options_update, specification
