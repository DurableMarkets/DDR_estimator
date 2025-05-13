import logit.ddr_tools.main_index as main_index
import logit.ddr_tools.regressors as regressors
import logit.monte_carlo_tools.simulate as simulate
import logit.monte_carlo_tools.misc_tools as misc_tools
import logit.monte_carlo_tools.monte_carlo_tools as monte_carlo_tools
import logit.ddr_tools.dependent_vars as dependent_vars
import logit.prices.prices as prices
import logit.estimators.pwls as pwls
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import jax
from tqdm import tqdm
import numpy as np
import eqb
from data_setups.options import get_model_specs

from set_path import get_paths
jax.config.update("jax_enable_x64", True)
pd.set_option("display.max_rows", 705)


path_dict = get_paths()
# load model options 

# Load jpe model
jpe_model = eqb.load_models("jpe_model")

### MODEL SPECIFICATION ###
sim_options, mc_options, params_update, options_update, specification, out_dir=get_model_specs(
    lambda model_name: f"./output/simulations/{model_name}/pwls/"
    )

# update
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
Nbars = mc_options['Nbars'] # if instead np.arange[0, est_size, x] we can do a sequentially increasing set of mc runs
ests, est_posts = [], []
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
    df_Nbar = df.groupby(["est_i", "consumer_type", "state", "decision"]).sum()

    chunks = df_Nbar.index.get_level_values("est_i").unique()[:mc_options['mc_iter']]
    # chunks = df.index.get_level_values("est_i").unique()[:mc_iter]
    # loops over monte carlo iterations
    for i in tqdm(chunks, desc="Current MC progress", leave=False):
        slicer = pd.IndexSlice[chunks[i : (i + 1)], :, :]
        # sim_df = df_Nbar.loc[slicer]

        # aggregate over chosen chunks
        sim_df = (
            df_Nbar.loc[slicer]
            .groupby(["consumer_type", "state", "decision"])
            .sum()
        ).reset_index()

        cfps, counts = dependent_vars.calculate_cfps_from_df(sim_df)
        #cfps= dependent_vars.true_ccps(main_df, model_solution, options)

        scrap_probabilities = dependent_vars.calculate_scrap_probabilities(sim_df)
        scrap_probabilities = model_solution["ccp_scrap_tau"]


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

        # Estimate 
        est, est_post = pwls.owls_regression_mc(
            ccps=cfps, 
            counts=counts, 
            X=X, 
            model_specification=model_specification
        )

        est = est.rename(columns={"Coefficient": "Estimates"})
        est["mc_iter"] = i
        est["Nbar"] = int(Nbar)
        est = est.set_index(["Nbar", "mc_iter"], append=True)
        
        ests.append(est)
        #breakpoint()
        est_post["mc_iter"] = i
        est_post["Nbar"] = int(Nbar)
        est_post = est_post.reset_index().set_index(
            ["Nbar", "mc_iter"]+
            ['consumer_type', 'decision', 'state', 
             'car_type_post_decision', 'car_age_post_decision'],
            append=True
        )
        est_posts.append(est_post)

# Extract true parameters
true_params=monte_carlo_tools.extract_true_structural_parameters(model_solution, model_struct_arrays, params, options)

true_params['mc_iter'] = 'true_ccps'
true_params.index = true_params.index.get_level_values(0)
true_params.index.name = "coefficients"

# combine runs
runs = pd.concat(ests, axis=0)

# Renaming indexes
runs.index.names = ["coefficients", "Nbars", "mc_iter"]

means = runs.groupby(["coefficients", "Nbars"], sort=False).mean()
means = means.rename(columns={"Estimates": "mean"})
list_of_Nbars = Nbars.tolist()

tuples = list(zip(["Sample size"] * len(list_of_Nbars), list_of_Nbars))
idx = pd.IndexSlice
for i, tuple in enumerate(tuples):
    means.loc[idx[tuple], "mean"] = list_of_Nbars[i]

tuples = list(zip(["MC iterations"] * len(list_of_Nbars), list_of_Nbars))
idx = pd.IndexSlice
for i, tuple in enumerate(tuples):
    means.loc[idx[tuple], "mean"] = mc_options['mc_iter']

# long table
stds = runs.groupby(["coefficients", "Nbars"], sort=False).std()
stds = stds.rename(columns={"Estimates": "std"})
p_025 = runs.groupby(["coefficients", "Nbars"], sort=False).quantile(0.025)
p_025 = p_025.rename(columns={"Estimates": "p2.5"})
p_975 = runs.groupby(["coefficients", "Nbars"], sort=False).quantile(0.975)
p_975 = p_975.rename(columns={"Estimates": "p97.5"})
# stats = pd.concat([true_est, means, stds, p_025, p_975], axis=1)
stats = pd.concat([means, stds, p_025, p_975], axis=1)
#
# # adding true values:
varnames = stats.index.get_level_values('coefficients').to_list()
tuples = list(zip(varnames, ["true values"] * len(varnames)))
true_params_df = true_params.reset_index()
true_params_df['Nbars'] = 'true value'
true_params_df = true_params_df.set_index(['coefficients', 'Nbars'])
true_params_df = true_params_df.rename(columns={'true values': 'mean'})
true_params_df = true_params_df[['mean']]
stats = pd.concat([stats, true_params_df], axis=0)

#idx = pd.IndexSlice
#for i, tuple in enumerate(tuples):
#    breakpoint()
#    stats.loc[idx[tuple], "mean"] = true_params['true values'].loc[tuple[0]]
#breakpoint()
#stats.index = stats.index.set_levels(
#    pd.Categorical(
#        stats.index.levels[0],
#        categories=true_params.index.tolist() + ["Sample size", "MC iterations"],
#        ordered=True,
#    ),
#    level=0,
#)
#stats = stats.sort_index(level=0)

stats = stats.sort_index(
    level=["coefficients", "Nbars"], ascending=[True, True]
).reset_index()


stats.round(4).to_latex(out_dir + "mc_table_long.tex", escape=False)
stats.round(4).to_markdown(out_dir + "mc_table_long.md")
# short table:
largest_Nbar = int(Nbars.max())
runs_largest = runs.loc[pd.IndexSlice[:, largest_Nbar, :], :].reset_index(
    level="Nbars", drop=True
)

means = runs_largest.groupby(["coefficients"], sort=False).mean()
means = means.rename(columns={"Estimates": "mean"})

means.loc["Sample size", "mean"] = largest_Nbar
means.loc["MC iterations", "mean"] = mc_options['mc_iter']


stds = runs_largest.groupby(["coefficients"], sort=False).std()
stds = stds.rename(columns={"Estimates": "std"})
p_025 = runs_largest.groupby(["coefficients"], sort=False).quantile(0.025)
p_025 = p_025.rename(columns={"Estimates": "p2.5"})
p_975 = runs_largest.groupby(["coefficients"], sort=False).quantile(0.975)
p_975 = p_975.rename(columns={"Estimates": "p97.5"})
stats = pd.concat([true_params, means, stds, p_025, p_975], axis=1)

stats.round(4).to_latex(out_dir + "mc_table.tex", escape=False)
stats.round(4).to_markdown(out_dir + "mc_table.md")


# plotting errors

est_post = pd.concat(est_posts, axis=0)
plt.figure(figsize=(25.6,14.4))
plt.scatter(est_post['ccps'], est_post['residuals'], marker='.', label='res', color='blue', alpha=0.5)
plt.xlabel("CCPs")
plt.ylabel("Residuals")
plt.legend()
plt.savefig(out_dir + "errors.png", dpi=100, bbox_inches="tight")


plt.figure(figsize=(25.6,14.4))
plt.scatter(est_post['counts'], est_post['residuals'], marker='.', label='res', color='blue', alpha=0.5)
plt.xlabel("counts")
plt.ylabel("Residuals")
plt.legend()
plt.savefig(out_dir + "errors_on_counts.png", dpi=100, bbox_inches="tight")


# Plotting them in a density plot
plt.figure(figsize=(25.6,14.4))
plt.hist(est_post['residuals'], bins=100, density=True, alpha=0.5, color='blue', label='res')
#plt.hist(np.exp(est_post['residuals']), bins=100, density=True, alpha=0.5, color='red', label='exp(res) - 1 ')
plt.title('optimal WLS residuals')
plt.xlabel("Residuals") 
plt.legend()
plt.savefig(out_dir + "errors_density.png", dpi=100, bbox_inches="tight")



