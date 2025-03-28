# %%
import sys

sys.path.insert(0, "../../src/")
from jpe_model.update_specs import update_params_and_options
import monte_carlo.mctools as mc
from eqb_model.process_model_struct import create_model_struct_arrays
from eqb_model.equilibrium import equilibrium_solver
import jax.numpy as jnp
import pandas as pd

from jax import config

config.update("jax_enable_x64", True)

# %%
# This file will be changed due to a refactor of the code base
# Rough outline is this:

# Set options
# store plots?
store_plots = True
hot_start = False
if hot_start:
    out_dir = "./results/uniform_ownership/"
    df_init_pname = "data_at_20231128-103111.pkl"


# Set number of consumers and car types
num_consumers = 1
num_car_types = 1

# number of observations used in simulation
chunk_size = 10000
sim_options = {
    "num_agents": chunk_size * 200,  # 226675,
    "num_periods": 1,
    "seed": 0,
    "chunk_size": chunk_size,
    "use_count_data": True,
}

# update options and params with number of consumers and car types
options_update = {
    "num_consumer_types": num_consumers,
    "num_car_types": num_car_types,
    "max_age_of_car_types": [25],
}

params_update = {
    "mum": [0.2],
    "p_fuel": [0.0],
    "acc_0": [-100],
    "tw": [1.0],
}

params, options = update_params_and_options(
    params=params_update, options=options_update
)


model_struct_arrays = create_model_struct_arrays(options=options)

# Solve the model
equ_output = equilibrium_solver(
    params=params,
    options=options,
    model_struct_arrays=model_struct_arrays,
)

# create a dict of prices
prices = mc.construct_price_dict(
    equ_price=equ_output["equilibrium_prices"],
    state_space_arrays=model_struct_arrays,
    params=params,
    options=options,
)

# Hack here to prevent extremely slow coverage of the state space
assert (
    equ_output["ownership_distribution_tau"].shape[0] == 1
), "This hack is not compatible with multiple consumer types"

nstates = model_struct_arrays["state_space"].shape[0]
# equ_output['ownership_distribution_tau'] = jnp.array(1*([1/nstates] * nstates)).reshape(1, nstates)

# Simulate data
df = mc.simulate_data(
    equ_output=equ_output,
    options=options,
    params=params,
    sim_options=sim_options,
    state_space_arrays=model_struct_arrays,
)
# If hot start is chosen load previous runs and add new simulations to this
if hot_start:
    df_init = pd.read_pickle(out_dir + df_init_pname)
    hot_start_i = df_init.index.get_level_values(0).max()
    new_index = df.index.get_level_values(0).unique() + hot_start_i + 1
    df.index = df.index.set_levels(new_index, level=0)
    df = pd.concat([df_init, df])


# %%

# Creating a checkpoint here:
# save_checkpoint = False
# if save_checkpoint:
#    out_dir = "./results/uniform_ownership/"
#    df.to_pickle(out_dir+"simulated_data_18bil_seed10000.pkl")

# %%

# Estimate clogit or pass frequency estimator on directly
true_ccps = mc.true_ccps(
    equ_output=equ_output,
)
ests = []
# Aggregate over chunks
# n_full_chunks, last_chunk_size = divmod(sim_options['num_agents'], sim_options['chunk_size'])
# chunk_sizes = [sim_options['chunk_size']]*(n_full_chunks)+ ([last_chunk_size] if last_chunk_size > 0 else [])

chunks = df.index.get_level_values("chunk_i").unique()
for i in chunks:
    slicer = pd.IndexSlice[chunks[: (i + 1)], :, :]
    sim_df = df.loc[slicer]

    # aggregate over chosen chunks
    sim_df = (
        sim_df.groupby(["consumer_type", "state_idx", "decision_idx"]).sum()
    ).reset_index()

    # Calc cfps
    cfps = mc.calculate_cfps(
        df=sim_df,
        state_decision_arrays=model_struct_arrays,
    )
    # Doing a quick hack by subtracting 2 since there is 2 illegal decision state pairs
    rows_dropped = jnp.isnan(cfps).sum() - 2
    # If visuals are desired, run the following:
    # mc.show_plots(true_ccps, est_ccps, cfps, state_decision_arrays=state_space_arrays)

    # Estimate DDR regression - ccps can be
    est = mc.DDR_regression(
        ccps=cfps,
        prices=prices,
        state_decision_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    est = est.rename(columns={"Coefficient1": "Estimates"})
    nobs = sim_options["chunk_size"] * (i + 1)  # sum(chunk_sizes[:(i+1)])
    est["obs"] = nobs
    est = est.set_index("obs", append=True)
    est.loc[pd.IndexSlice["rows_dropped", nobs], "Estimates"] = rows_dropped

    ests.append(est)

# Do a final estimation on the true ccps
true_est = mc.DDR_regression(
    ccps=true_ccps,
    prices=prices,
    state_decision_arrays=model_struct_arrays,
    params=params,
    options=options,
)
true_est = true_est.rename(columns={"Coefficient1": "Estimates"})
true_est["obs"] = "true_ccps"
true_est = true_est.set_index("obs", append=True)
ests.append(true_est)


# combine runs
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
runs = pd.concat(ests, axis=0)

# store results:
out_dir = "./results/uniform_ownership/"

runs.to_pickle(out_dir + "runs_at_" + timestamp + ".pkl")
df.to_pickle(out_dir + "data_at_" + timestamp + ".pkl")

# Plot results

# %%
import pandas as pd
import matplotlib.pyplot as plt

out_dir = "./results/uniform_ownership/"
runs = pd.read_pickle(out_dir + "runs_at_" + timestamp + ".pkl")

coeff = "price_1"
est_values = runs.loc[coeff].values[:-1] / runs.loc["car_0_1"].values[:-1]
true = runs.loc[coeff].values[-1] / runs.loc["car_0_1"].values[-1]
x = runs.index.get_level_values("obs").unique()[:-1]
row_changes = runs.loc[pd.IndexSlice[coeff]][:-1][
    (
        runs.loc["rows_dropped", "Estimates"]
        - runs.loc["rows_dropped", "Estimates"].shift(1)
        < 0
    )
].index.values
plt.plot(x, est_values)
plt.axhline(y=true, color="red")
for vert in row_changes:
    plt.axvline(x=vert, color="green")


if store_plots:
    plt.savefig(out_dir + "asymptotics_mum.png")
    plt.show()
    mc.show_plots(
        true_ccps,
        cfps,
        cfps,
        state_decision_arrays=model_struct_arrays,
        save_to=out_dir,
    )
else:
    plt.show()
    mc.show_plots(true_ccps, cfps, cfps, state_decision_arrays=model_struct_arrays)

breakpoint()

# %%
