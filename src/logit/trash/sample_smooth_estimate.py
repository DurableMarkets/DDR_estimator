import sys

sys.path.insert(0, "../../src/")
import pdb
from jpe_model.model_specs import set_options
from jpe_model.model_specs import set_params
from jpe_model.model_specs import update_params_by_options
import monte_carlo.mctools as mc
from eqb_model.process_model_space import create_state_and_decision_space_objects
from eqb_model.equlibrium import create_equilbrium_outputs
from eqb_model.equlibrium import equilibrium_solver
import jax.numpy as jnp

# This file will be changed due to a refactor of the code base
# Rough outline is this:

# Set options

# Set number of consumers and car types
num_consumers = 1
num_car_types = 1

# number of observations used in simulation
sim_options = {
    "num_agents": 1000000,  # 226675,
    "num_periods": 1,
    "seed": 123,
    "chunk_size": 1000000,
    "use_count_data": True,
}

# update options and params with number of consumers and car types
params = set_params()
options = set_options()
options = options._replace(num_consumer_types=num_consumers)
options = options._replace(num_car_types=num_car_types)
options = options._replace(max_age_of_car_types=jnp.array([25] * num_car_types))

params = update_params_by_options(params, options)


state_space_arrays = create_state_and_decision_space_objects(options=options)

# Solve the model
equ_price = equilibrium_solver(
    params=params,
    options=options,
    state_decision_space_arrays=state_space_arrays,
)

equ_output = create_equilbrium_outputs(
    equ_prices=equ_price,
    params=params,
    options=options,
    state_decision_space_arrays=state_space_arrays,
)

# create a dict of prices
prices = mc.construct_price_dict(
    equ_price=equ_price,
    state_space_arrays=state_space_arrays,
    params=params,
    options=options,
)

# Simulate data
sim_df = mc.simulate_data(
    equ_output=equ_output,
    options=options,
    params=params,
    sim_options=sim_options,
    state_space_arrays=state_space_arrays,
)

# Aggregate over chunks
sim_df = (
    sim_df.groupby(["consumer_type", "state_idx", "decision_idx"]).sum()
).reset_index()

# Estimate clogit or pass frequency estimator on directly

est_options = {
    "w_j_vars": ["age_pol", "new", "d1", "trade", "keep"],
    "s_i_vars": ["age_pol", "nocar"],
    "w_j_deg": 5,
    "s_i_deg": 5,
    "unwanted_choices": None,
}

# True ccps are the same as in equ_output['ccps_tau'] - assuming one consumer type
# cfps could be calculated independently of this function such that it would only return the smoothed ccps and a result module
true_ccps = mc.true_ccps(
    equ_output=equ_output,
)

cfps = mc.calculate_cfps(
    df=sim_df,
    state_decision_arrays=state_space_arrays,
)


smoothing = True
if smoothing:
    est_ccps = mc.estimate_clogit(
        df=sim_df,
        state_decision_arrays=state_space_arrays,
        equ_output=equ_output,
        **est_options,
    )

# If visuals are desired, run the following:
mc.show_plots(true_ccps, est_ccps, cfps, state_decision_arrays=state_space_arrays)

# Estimate DDR regression - ccps can be
est = mc.DDR_regression(
    ccps=cfps,
    prices=prices,
    state_decision_arrays=state_space_arrays,
    params=params,
    options=options,
)

print(est)
# breakpoint()

# Within each step data is returned from functions and therefore kept in memory. This should significantly speed up the process.

# let's go
