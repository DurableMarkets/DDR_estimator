import logit.ddr_tools.main_index as main_index
import logit.ddr_tools.regressors as regressors
import logit.monte_carlo_tools.simulate as simulate
import logit.monte_carlo_tools.misc_tools as misc_tools
import logit.monte_carlo_tools.monte_carlo_tools as monte_carlo_tools
import logit.ddr_tools.dependent_vars as dependent_vars
# from eqb.process_model_struct import create_model_struct_arrays
# from eqb.equilibrium import equilibrium_solver
# import DDR_estimation.utils
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import jax
from tqdm import tqdm
import numpy as np
import eqb

from set_path import get_paths

path_dict = get_paths()

# Load jpe model
jpe_model = eqb.load_models("jpe_model")

jax.config.update("jax_enable_x64", True)
pd.set_option("display.max_rows", 705)

# set output directory
out_dir = "./results/wls monte carlo runs/linear specification/test/"

# Set options

# Set number of consumers and car types
num_consumers = 2
num_car_types = 2

### MODEL SPECIFICATION ###

specification = {
    "mum": (num_consumers, 1),
    "buying": (num_consumers, 1),
    "scrap_correction": (num_consumers, 1),
    "u_0": (num_consumers, 1),
    "u_a": (num_consumers, num_car_types),
    "u_a_sq": None,
    "u_a_even": None,
}

# chunk_size and n_periods should be tuned to jax's memory capacity and mc_iter should control the number of observations.
chunk_size = 100_000
mc_iter = 100
N_mc = 500_000 #5_000_000 
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
prices = misc_tools.construct_price_dict(
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
    prices=prices,
    model_struct_arrays=model_struct_arrays,
    model_funcs = jpe_model,
    params=params,
    options=options,
    specification=specification,
)


# mc simulation
ests = []
df_idx = df.index
#TODO: Strange that the indices are named differently across the data df and the main_df...
# Loops over monte carlos of different sizes
for Nbar in tqdm(Nbars, desc="Monte Carlo studies"):
    # redefining sim_options
    sim_options["estimation_size"] = Nbar

    # replace sim_index with with estimation index.
    new_index = monte_carlo_tools.update_sim_index_to_est_index(df_idx, sim_options)
    # This changes the index of df to the estimation index
    # in each loop I overwrite existing index in the same dataframe df
    df.index = new_index

    # aggregate over new index
    df_Nbar = df.groupby(["est_i", "consumer_type", "state_idx", "decision_idx"]).sum()

    chunks = df_Nbar.index.get_level_values("est_i").unique()[:mc_iter]
    # chunks = df.index.get_level_values("est_i").unique()[:mc_iter]
    # loops over monte carlo iterations
    for i in tqdm(chunks, desc="Current MC progress", leave=False):
        slicer = pd.IndexSlice[chunks[i : (i + 1)], :, :]
        # sim_df = df_Nbar.loc[slicer]

        # aggregate over chosen chunks
        sim_df = (
            df_Nbar.loc[slicer]
            .groupby(["consumer_type", "state_idx", "decision_idx"])
            .sum()
        ).reset_index()

        cfps, counts = dependent_vars.calculate_cfps_from_df(sim_df)

        
        cfps = cfps.loc[I_feasible, :]
        counts = counts.loc[I_feasible, :]

        scrap_probabilities = dependent_vars.calculate_scrap_probabilities(sim_df)
#
#         # Estimate accident parameters
#         #acc_0_hat = mc.calculate_accident_parameters(scrap_probabilities=scrap_probabilities)
#
#         # Update params to acomodate new "estimated" acc_0
#         #params_hat, options = update_params_and_options(params={"acc_0": acc_0_hat}, options=options)
#         params_hat = params
#
#         # create data dependent regressors
#         X_dep, _ = utils.create_data_dependent_regressors(
#             tab_index=tab_index,
#             prices=prices,
#             scrap_probabilities=scrap_probabilities,
#             state_decision_arrays=model_struct_arrays,
#             params=params_hat,
#             options=options,
#             specification=specification,
#         )
#
#         # combine independent and dependent regressors
#         X = utils.create_regressors_combine_parts(X_indep, X_dep, model_specification)
#
#         # Estimate pddr regression
#         est = mc.wls_regression_mc(
#             ccps=cfps,
#             counts=counts,
#             X=X,
#             model_specification=model_specification,
#         )
#
#         est = est.rename(columns={"Coefficient": "Estimates"})
#         est["mc_iter"] = i
#         est["Nbar"] = int(Nbar)
#         est = est.set_index(["Nbar", "mc_iter"], append=True)
#
#         ests.append(est)
#
# # Extract true parameters
# true_params=mc.extract_true_structural_parameters(equ_output, model_struct_arrays, params, options)
#
# true_params['mc_iter'] = 'true_ccps'
# true_params.index = true_params.index.get_level_values(0)
# true_params.index.name = "coefficients"
#
# breakpoint()
# # combine runs
# runs = pd.concat(ests, axis=0)
#
# # Renaming indexes
# runs.index.names = ["coefficients", "Nbars", "mc_iter"]
#
# # Build a table that demonstrates the properties of the PDDR.
# # I would liketo include mean value and standard deviation of the estimates,
# # As a minimum.
#
# means = runs.groupby(["coefficients", "Nbars"], sort=False).mean()
# means = means.rename(columns={"Estimates": "mean"})
# list_of_Nbars = Nbars.tolist()
#
# tuples = list(zip(["Sample size"] * len(list_of_Nbars), list_of_Nbars))
# idx = pd.IndexSlice
# for i, tuple in enumerate(tuples):
#     means.loc[idx[tuple], "mean"] = list_of_Nbars[i]
#
# tuples = list(zip(["MC iterations"] * len(list_of_Nbars), list_of_Nbars))
# idx = pd.IndexSlice
# for i, tuple in enumerate(tuples):
#     means.loc[idx[tuple], "mean"] = mc_iter
#
# # long table
# stds = runs.groupby(["coefficients", "Nbars"], sort=False).std()
# stds = stds.rename(columns={"Estimates": "std"})
# p_025 = runs.groupby(["coefficients", "Nbars"], sort=False).quantile(0.025)
# p_025 = p_025.rename(columns={"Estimates": "p2.5"})
# p_975 = runs.groupby(["coefficients", "Nbars"], sort=False).quantile(0.975)
# p_975 = p_975.rename(columns={"Estimates": "p97.5"})
# # stats = pd.concat([true_est, means, stds, p_025, p_975], axis=1)
# stats = pd.concat([means, stds, p_025, p_975], axis=1)
#
# # adding true values:
# varnames = stats.index.get_level_values('coefficients').to_list()
# tuples = list(zip(varnames, ["true values"] * len(varnames)))
# true_params_df = true_params.reset_index()
# true_params_df['Nbars'] = 'true value'
# true_params_df = true_params_df.set_index(['coefficients', 'Nbars'])
# true_params_df = true_params_df.rename(columns={'true values': 'mean'})
# true_params_df = true_params_df[['mean']]
# stats = pd.concat([stats, true_params_df], axis=0)
#
# #idx = pd.IndexSlice
# #for i, tuple in enumerate(tuples):
# #    breakpoint()
# #    stats.loc[idx[tuple], "mean"] = true_params['true values'].loc[tuple[0]]
# #breakpoint()
# #stats.index = stats.index.set_levels(
# #    pd.Categorical(
# #        stats.index.levels[0],
# #        categories=true_params.index.tolist() + ["Sample size", "MC iterations"],
# #        ordered=True,
# #    ),
# #    level=0,
# #)
# #stats = stats.sort_index(level=0)
#
# stats = stats.sort_index(
#     level=["coefficients", "Nbars"], ascending=[True, True]
# ).reset_index()
#
#
# stats.round(4).to_latex(out_dir + "mc_table_long.tex", escape=False)
# stats.round(4).to_markdown(out_dir + "mc_table_long.md")
# # short table:
# largest_Nbar = int(Nbars.max())
# runs_largest = runs.loc[pd.IndexSlice[:, largest_Nbar, :], :].reset_index(
#     level="Nbars", drop=True
# )
#
# means = runs_largest.groupby(["coefficients"], sort=False).mean()
# means = means.rename(columns={"Estimates": "mean"})
#
# means.loc["Sample size", "mean"] = largest_Nbar
# means.loc["MC iterations", "mean"] = mc_iter
#
#
# stds = runs_largest.groupby(["coefficients"], sort=False).std()
# stds = stds.rename(columns={"Estimates": "std"})
# p_025 = runs_largest.groupby(["coefficients"], sort=False).quantile(0.025)
# p_025 = p_025.rename(columns={"Estimates": "p2.5"})
# p_975 = runs_largest.groupby(["coefficients"], sort=False).quantile(0.975)
# p_975 = p_975.rename(columns={"Estimates": "p97.5"})
# stats = pd.concat([true_params, means, stds, p_025, p_975], axis=1)
#
# stats.round(4).to_latex(out_dir + "mc_table.tex", escape=False)
# stats.round(4).to_markdown(out_dir + "mc_table.md")
#
#
# # Build a violin plot of the estimates for each parameter.
# ## Skipping this.
# #import seaborn as sns
# #import matplotlib.patches as mpatches
#
# #plt.figure(figsize=(20, 12))
#
# #diffs = runs["Estimates"] - true_params["true values"]
#
# #colors = sns.color_palette("Set1", n_colors=len(Nbars))
# #patches = []
# #for i, Nbar in enumerate(Nbars):
# #    diffs_n = diffs.loc[pd.IndexSlice[:, int(Nbar), :]]
# #    sns.boxplot(
# #        x=diffs_n.values,
# #        y=diffs_n.index.get_level_values("coefficients"),
# #        orient="h",
# #        boxprops=dict(alpha=0.3, color=colors[i]),
# #        showfliers=False,
# #        fill=False,
# #        legend=True,
# #    )
# #    patches.append(
# #        mpatches.Patch(
# #            color=colors[i], label="{:0.1e}".format(Nbar.astype(int)) + " Obs."
# #        )
# #    )
# #
# #plt.legend(handles=patches)
#
# #plt.savefig(out_dir + "violin_plot.png", dpi=300, bbox_inches="tight")
#
# # plt.show()
