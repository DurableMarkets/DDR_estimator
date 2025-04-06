import os
import numpy as np
import pandas as pd
import pickle
from logit.ddr_tools.main_index import (
    create_main_feasible_idx,
)

import jax
import jax.numpy as jnp

TEST_RESOURCES_MODEL_1 = "../../tests/resources/simple_model/matlab_files/"

from eqb.equilibrium import (
    create_model_struct_arrays,
    equilibrium_solver,
)
from eqb.simulate import simulate_with_solution_jax

jax.config.update("jax_enable_x64", True)

def load_or_simulate_data(params, options, model_funcs, sim_options, datadir):
    # search for existing datasets
    files = os.listdir(datadir)

    # if nfiles is 0 then loop is skipped:
    nfiles = len(files)

    checks_passed = False
    for f in files:
        if nfiles == 0:
            break

        # file path
        fpath = os.path.join(datadir, f)

        # load data
        with open(fpath, "rb") as f:
            (
                fmodel_solution,
                fdf,
                fparams,
                foptions,
                fsim_options,
            ) = pickle.load(f)

        # check if params and options matches existing data
        params_to_check = set(params.keys()) & set(fparams.keys())
        try:
            pcheck_result = jnp.all(
                jnp.array(
                    [jnp.all(fparams[par] == params[par]) for par in params_to_check]
                )
            )
        except:
            pcheck_result = False

        options_to_check = set(options.keys()) & set(foptions.keys())
        try:
            ocheck_result = jnp.all(
                jnp.array(
                    [jnp.all(foptions[opt] == options[opt]) for opt in options_to_check]
                )
            )
        except:
            ocheck_result = False

        same_model_check = pcheck_result & ocheck_result

        # check if estimation_size is divisible with chunk_size - Otherwise aggregation cannot be done.
        divisible_check = (
            sim_options["estimation_size"] % fsim_options["chunk_size"] == 0
        )

        # check if n_agents  are the same:
        sim_check = np.all(
            np.array([fsim_options[opt] == sim_options[opt] for opt in ["n_periods"]])
        )

        # check if the maximum number of observations are in the existing dataset.
        size_check = (
            sim_options["n_agents"] * sim_options["n_periods"]
            <= fsim_options["n_agents"] * fsim_options["n_periods"]
        )

        checks_passed = same_model_check & divisible_check & sim_check & size_check

        if checks_passed:
            break
        else:
            continue

    # read the data or simulate new data
    if checks_passed:
        print(
            "Found a file with matching parameters and options. You requested {:,.0f} thousand obs. and file contain {:,.0f} thousand obs.".format(
                sim_options["n_agents"] * sim_options["n_periods"] / 1000,
                fsim_options["n_agents"] * fsim_options["n_periods"] / 1000,
            )
        )

        print("Loading data...")
        model_solution = fmodel_solution
        df = fdf
        params = fparams
        options = foptions
        sim_options = fsim_options
        model_struct_arrays=create_model_struct_arrays(options, model_funcs)
        pass  # data is already loaded
    else:
        print("No file with matching parameters and options found. Simulating data...")
        (
            model_solution,
            df,
            params,
            options,
            model_struct_arrays,
        ) = solve_and_simulate_data_jax(params, options, model_funcs, sim_options)

        # renaming multiindex levels of df
        df.rename_axis(
            index=
                {
                #"consumer_type": "tau", 
                "state_idx": "state",
                "decision_idx": "decision"
                },
        inplace=True)

        # removing infeasible state, decision pairs. 
        feasible_idx = create_main_feasible_idx(
            model_struct_arrays=model_struct_arrays,
            params=params,
            options=options,
        )

        feasible_idx = feasible_idx.reorder_levels(['consumer_type', 'state', 'decision'])

        df=df.loc[df.index.droplevel('chunk_i').isin(feasible_idx)]


        # Save the data
        print("Simulation done. Dumping data now..")
        with open(datadir + "data.pkl", "wb") as f:
            pickle.dump(
                (model_solution, df, params, options, sim_options), f
            )

    return model_solution, df, params, options, sim_options, model_struct_arrays


def solve_and_simulate_data_jax(params, options, model_funcs, sim_options):
    # seems redundant?
    params, options=model_funcs['update_params_and_options'](params=params, options=options)
    model_struct_arrays = create_model_struct_arrays(options=options, model_funcs=model_funcs)


    # Solve the model
    model_solution = jax.jit(
        lambda params_dict: equilibrium_solver(
            params=params_dict,
            options=options,
            model_struct_arrays=model_struct_arrays,
            model_funcs = model_funcs,
        )
    )(
        params,
    )

    # Simulate data
    df = simulate_data_jax(
        model_solution=model_solution, 
        model_struct_arrays=model_struct_arrays, 
        model_funcs=model_funcs, 
        params=params, 
        options=options, 
        sim_options=sim_options,
    )

    return model_solution, df, params, options, model_struct_arrays

def simulate_data_jax(
        model_solution, 
        model_struct_arrays, 
        model_funcs, 
        params, 
        options, 
        sim_options
):
    # Does sampling runs to reach the number of requested observations,
    # by chunking the sampling runs into smaller chunks of chunk_size.
    # chunk_size has to be divisible by the number of requested observations

    n_agents = sim_options["n_agents"]
    chunk_size = sim_options["chunk_size"]

    n_full_chunks, last_chunk_size = divmod(n_agents, chunk_size)
    if last_chunk_size > 0:
        raise Exception(
            "the number of agents times observations has to be divisible by chunk_size"
        )

    nseeds = n_full_chunks
    seeds = np.arange(sim_options["seed"], nseeds + sim_options["seed"])

    # jax.jit(simulate_with_solution,) # This would be the way to jit it but I
    # would not work with because sim_options_chunk is one argument.
    sim_options_chunk = {
        "n_agents": chunk_size,
        "n_periods": sim_options["n_periods"],
    }

    dfs = []

    # unpack options
    num_consumer_types = options["n_consumer_types"]
    num_car_types = options["n_car_types"]
    max_age_of_car_types = jnp.array(options["max_age_of_car_types"])
    num_states = model_struct_arrays["state_space"].shape[0]
    num_decisions = model_struct_arrays["decision_space"].shape[0]
    ncells = num_consumer_types * num_states * num_decisions
    n_shape = (num_consumer_types, num_states, num_decisions)

    # Creating indexer
    indexer = jnp.arange(ncells).reshape(n_shape)

    # creating columns
    consumer_idxs = np.full(
        n_shape,
        -9999,
    )

    for i in range(num_consumer_types):
        consumer_idxs[i, :, :] = i
    consumer_idxs = jnp.array(consumer_idxs).flatten()

    state_idxs = np.full(
        n_shape,
        -9999,
    )
    for s in range(num_states):
        state_idxs[:, s, :] = s
    state_idxs = jnp.array(state_idxs).flatten()

    decision_idxs = np.full(
        n_shape,
        -9999,
    )
    for d in range(num_decisions):
        decision_idxs[:, :, d] = d
    decision_idxs = jnp.array(decision_idxs).flatten()

    partial_simulate_chunk_jax = lambda seed: simulate_chunk_jax(
        model_solution=model_solution,
        sim_options=sim_options,
        sim_options_chunk=sim_options_chunk,
        model_struct_arrays=model_struct_arrays,
        model_funcs = model_funcs,
        seed=seed,
        indexer=indexer,
        consumer_idxs=consumer_idxs,
        state_idxs=state_idxs,
        decision_idxs=decision_idxs,
        ncells=ncells,
    )

    jit_simulate_chunk_jax = jax.jit(partial_simulate_chunk_jax)

    # This method seems to be faster
    dfs = jax.lax.map(jit_simulate_chunk_jax, xs=seeds)

    # combining chunks into one df
    dfs = jnp.reshape(dfs, (n_full_chunks * ncells, 6))

    df = pd.DataFrame(
        dfs,
        columns=[
            "chunk_i",
            "consumer_type",
            "state_idx",
            "decision_idx",
            "counts",
            "scrap_counts",
        ],
    ).set_index(["chunk_i", "consumer_type", "state_idx", "decision_idx"])

    # remove indices that are not valid

    return df


def simulate_chunk_jax(
    model_solution,
    sim_options,
    sim_options_chunk,
    model_struct_arrays,
    model_funcs,
    seed,
    indexer,
    consumer_idxs,
    state_idxs,
    decision_idxs,
    ncells,
):
    partial_sim = lambda seed: simulate_with_solution_jax(
        model_solution=model_solution,
        sim_options=sim_options_chunk,
        model_struct_arrays=model_struct_arrays,
        model_funcs=model_funcs,
        seed=seed,
    )
    # jit_sim = jax.jit(partial_sim)

    # Jitting obs_to_counts_jax
    partial_obs_to_counts_jax = lambda chunk_i, sim_dict: obs_to_counts_jax(
        sim_dict, chunk_i, ncells, indexer, consumer_idxs, state_idxs, decision_idxs
    )
    # jit_obs_to_counts=jax.jit(partial_obs_to_counts_jax)

    # simulate chunk of data:
    sim_dict = partial_sim(seed=seed)
    chunk_i = seed - sim_options["seed"]
    sim_data = partial_obs_to_counts_jax(chunk_i, sim_dict)

    return sim_data


def obs_to_counts_jax(
    sim_dict, i, ncells, indexer, consumer_idxs, state_idxs, decision_idxs
):
    sim_index = jnp.array(
        [sim_dict["consumer_type"], sim_dict["state_idx"], sim_dict["decision_idx"]]
    )

    # Columns: chunk_i, consumer_type, state_idx, decision_idx, counts
    chunk_is = jnp.repeat(i, ncells)
    # alternative could be jnp bincount across indexer
    # counts=jop.segment_sum(jnp.ones(chunk_size, dtype='int'), indexer[sim_index[0], sim_index[1], sim_index[2]])
    counts = jnp.bincount(
        indexer[sim_index[0], sim_index[1], sim_index[2]],
        minlength=ncells,
        length=ncells,
    )

    #
    valid_scrap_decisions_lookup = jnp.array([0, 1])
    valid_scrap_decisions = valid_scrap_decisions_lookup[sim_dict["scrap_decision"]]

    scrap_counts = jnp.bincount(
        indexer[sim_index[0], sim_index[1], sim_index[2]],
        weights=valid_scrap_decisions,
        minlength=ncells,
        length=ncells,
    )

    sim_data = jnp.column_stack(
        (chunk_is, consumer_idxs, state_idxs, decision_idxs, counts, scrap_counts)
    )

    return sim_data
