import logit.ddr_tools.main_index as main_index
import logit.ddr_tools.regressors as regressors
import logit.monte_carlo_tools.simulate as simulate
import logit.monte_carlo_tools.misc_tools as misc_tools
import logit.monte_carlo_tools.monte_carlo_tools as monte_carlo_tools
import logit.ddr_tools.dependent_vars as dependent_vars
import logit.prices.prices as prices
import logit.estimators.wls_estimation as wls_estimation
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import jax
from tqdm import tqdm
import numpy as np
import eqb
import logit.estimators.optimal_wls as optimal_wls
from set_path import get_paths

path_dict = get_paths()

# Load jpe model
jpe_model = eqb.load_models("jpe_model")

jax.config.update("jax_enable_x64", True)
pd.set_option("display.max_rows", 705)

# set output directory
out_dir = "./output/simulations/wls/"

# Set options

# Set number of consumers and car types
num_consumers = 2
num_car_types = 2

### MODEL SPECIFICATION ###

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

# chunk_size and n_periods should be tuned to jax's memory capacity and mc_iter should control the number of observations.
chunk_size = 500_000
mc_iter = 100
N_mc = 1_000_000 #5_000_000 
sample_iter = N_mc * mc_iter // chunk_size

# Estimation_size controls the sample size used in the estimation
estimation_size = N_mc  # 1000000
assert (
    estimation_size % chunk_size == 0
), "estimation_size should be a multiple of chunk_size"
assert N_mc % chunk_size == 0, "N_mc should be a multiple of chunk_size"
assert (
    estimation_size <= chunk_size * sample_iter
), "estimation_size should be smaller or equal to chunk_size * mc_iter"

sim_options = {
    "n_agents": chunk_size * sample_iter,  # 226675,
    "n_periods": 1,
    "seed": 123,
    "chunk_size": chunk_size,
    "estimation_size": estimation_size,
    "use_count_data": True,
}

# stores different sample sizes for multiple monte carlo runs
Nbars = jnp.arange(0, N_mc, 5 * 10**5) + 5 * 10**5
Nbars = jnp.array([N_mc])

# update options and params with number of consumers and car types
params_update = {
    "p_fuel": [0.0],
    "acc_0": [-100.0],
    "mum": [0.5, 0.5],
    "psych_transcost": [4.0, 2.0],
    'u_0': np.array([[12.0,12.0],[12.0,12.0]]),
    #'u_a': np.array([-0.5,-0.5]),
    #'sigma_sell_scrapp': 0.0000000001,
    #'pscrap': [1.0,1.0],
}

options_update = {
    "n_consumer_types": num_consumers, # Redundant
    "n_car_types": num_car_types,
    "max_age_of_car_types": [25],
    "tw": [0.5, 0.5],
}
params, options = jpe_model["update_params_and_options"](
    params=params_update, options=options_update
)

# Simulate data or load data
(
    model_solution,
    df,
    params,
    options,
    fsim_options,
    model_struct_arrays,
) = simulate.load_or_simulate_data(params, options, jpe_model, sim_options, path_dict["sim_data"])


for key in model_struct_arrays.keys():
    model_struct_arrays[key] = np.array(model_struct_arrays[key])

# # create a dict of prices
price_dict = misc_tools.construct_price_dict(
    equ_price=model_solution["equ_prices"],
    state_space_arrays=model_struct_arrays,
    params=params,
    options=options,
)

# Create main index used for constructing regressors
main_df = main_index.create_main_df(
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
)

# creating data independent regressors
X_indep, model_specification = regressors.create_data_independent_regressors(
    main_df=main_df,
    prices=price_dict,
    model_struct_arrays=model_struct_arrays,
    model_funcs = jpe_model,
    params=params,
    options=options,
    specification=specification,
)


# mc simulation
ests = []
df_idx = df.index
sim_options["estimation_size"] = Nbars[0]  # set estimation size to Nbar

new_index = monte_carlo_tools.update_sim_index_to_est_index(df_idx, sim_options)
# This changes the index of df to the estimation index
# in each loop I overwrite existing index in the same dataframe df
df.index = new_index

# aggregate over new index
df_Nbar = df.groupby(["est_i", "consumer_type", "state", "decision"]).sum()

chunks = df_Nbar.index.get_level_values("est_i").unique()[:mc_iter]
i = chunks[0]
slicer = pd.IndexSlice[chunks[i : (i + 1)], :, :]
# sim_df = df_Nbar.loc[slicer]

# aggregate over chosen chunks
sim_df = (
    df_Nbar.loc[slicer]
    .groupby(["consumer_type", "state", "decision"])
    .sum()
).reset_index()

cfps, counts = dependent_vars.calculate_cfps_from_df(sim_df)
#cfps = dependent_vars.true_ccps(main_df, model_solution, options)

scrap_probabilities = dependent_vars.calculate_scrap_probabilities(sim_df)
#scrap_probabilities = model_solution["ccp_scrap_tau"]


# Estimate accident parameters
#acc_0_hat = monte_carlo_tools.calculate_accident_parameters(scrap_probabilities=scrap_probabilities)

# Update params to acomodate new "estimated" acc_0
#params_hat, options = update_params_and_options(params={"acc_0": acc_0_hat}, options=options)
#params_hat = params

# create data dependent regressors
X_dep, _ = prices.create_data_dependent_regressors(
    main_df=main_df,
    prices=price_dict,
    scrap_probabilities=scrap_probabilities,
    model_struct_arrays=model_struct_arrays,
    model_funcs=jpe_model,
    params=params,
    options=options,
    specification=specification,
)
# combine independent and dependent regressors
X = dependent_vars.combine_regressors(X_indep, X_dep, model_specification)

#optimal_wls_weights = optimal_wls.calculate_weights(ccps=cfps, counts=counts)

optimal_wls.test_of_covariance_matrices(X, cfps, counts)
optimal_wls.playground_test_of_pseudo_inverses(
    ccps=cfps,
    counts=counts, 
    #X=X, 
    #model_specification=model_specification
    )


oest=optimal_wls.owls_regression_mc(ccps=cfps, counts=counts, X=X, model_specification=model_specification)
west=wls_estimation.wls_regression_mc(ccps=cfps, counts=counts, X=X, model_specification=model_specification)


