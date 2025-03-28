# %%
import sys

sys.path.insert(0, "../../src/")
from jpe_model.update_specs import update_params_and_options
import monte_carlo.mctools as mc
from eqb_model.process_model_struct import create_model_struct_arrays
from eqb_model.equilibrium import equilibrium_solver
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    "num_agents": chunk_size * 100,  # 226675,
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

if hot_start:
    df = pd.read_pickle(out_dir + df_init_pname)

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
equ_output["ownership_distribution_tau"] = jnp.array(
    1 * ([1 / nstates] * nstates)
).reshape(1, nstates)

# Simulate data
df = mc.simulate_data(
    equ_output=equ_output,
    options=options,
    params=params,
    sim_options=sim_options,
    state_space_arrays=model_struct_arrays,
)

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

chunks = df.index.get_level_values("chunk_i").unique()

cfpss = []
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

    cfpss.append(cfps)

cfpss = np.array(cfpss)

# estimates
low_prob_cfp = cfpss[:, 0, -3]
high_prob_cfp = cfpss[:, 0, 0]

# true
low_prob_true = true_ccps[0, -3]
high_prob_true = true_ccps[0, 0]

# getting the x axis:
x = df.loc[pd.IndexSlice[:, 0, 0, :]].groupby("chunk_i").sum().cumsum().values


plt.subplot(1, 2, 1)
plt.plot(x, low_prob_cfp, label="low prob cfp", color="black", linestyle="--")
plt.plot(x, high_prob_cfp, label="high prob cfp", color="red", linestyle="--")

plt.axhline(
    low_prob_true,
    label="low prob true",
    color="black",
)
plt.axhline(
    high_prob_true,
    label="high prob true",
    color="red",
)
plt.legend()
plt.xlabel("Number of observations")
plt.ylabel("Cond. Choice frequency")


plt.subplot(1, 2, 2)
plt.plot(x, np.log(low_prob_cfp), label="low prob cfp", color="black", linestyle="--")
plt.plot(x, np.log(high_prob_cfp), label="high prob cfp", color="red", linestyle="--")
plt.xlabel("Number of observations")
plt.ylabel("Log Cond. choice frequency")

plt.axhline(
    np.log(low_prob_true),
    label="low prob true ({:0.4f} %)".format(low_prob_true * 100),
    color="black",
)
plt.axhline(
    np.log(high_prob_true),
    label="high prob true({:0.4f} %)".format(high_prob_true * 100),
    color="red",
)
plt.legend()
plt.show()


breakpoint()
