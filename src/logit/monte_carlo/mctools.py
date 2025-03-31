import sys

import jax.numpy as jnp
import jax.ops as jop

sys.path.insert(0, "../../src/")
sys.path.insert(0, "./DDR_estimation/")
sys.path.insert(0, "./monte_carlo/")

TEST_RESOURCES_MODEL_1 = "../../tests/resources/simple_model/matlab_files/"
import eqb
import logit.DDR_estimation.model_interface as mi
import logit.DDR_estimation.pdr_estimation as pdr
import logit.DDR_estimation.wls_estimation as wls
import logit.DDR_estimation.utils
from scipy.io import loadmat
import jax
from tqdm import tqdm

import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import pandas as pd
from logit.monte_carlo.clogit import ccp
from logit.monte_carlo.clogit import clogit
from logit.monte_carlo.clogit import utility
from logit.monte_carlo.generate_explanatory_variables import create_x_matrix

from eqb.equilibrium import (
     create_structure_and_solve_model,
     create_model_struct_arrays,
     equilibrium_solver
)
jax.config.update("jax_enable_x64", True)


def construct_price_dict(equ_price, state_space_arrays, params, options):
    np_state_decision_arrays = jax.tree_util.tree_map(
        lambda x: np.array(x), state_space_arrays
    )

    prices = {
        "used_car_prices": np.array(equ_price),
        "new_car_prices": np.array(params["pnew"]),
        "scrap_car_price": np.array(params["pscrap"]),
        "used_prices_indexer": np_state_decision_arrays["map_state_to_price_index"],
    }

    return prices


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
                fequ_output,
                fdf,
                fparams,
                foptions,
                fsim_options,
                fmodel_struct_arrays,
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

        # check if num_agents  are the same:
        sim_check = np.all(
            np.array([fsim_options[opt] == sim_options[opt] for opt in ["num_periods"]])
        )

        # check if the maximum number of observations are in the existing dataset.
        size_check = (
            sim_options["num_agents"] * sim_options["num_periods"]
            <= fsim_options["num_agents"] * fsim_options["num_periods"]
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
                sim_options["num_agents"] * sim_options["num_periods"] / 1000,
                fsim_options["num_agents"] * fsim_options["num_periods"] / 1000,
            )
        )

        print("Loading data...")
        equ_output = fequ_output
        df = fdf
        params = fparams
        options = foptions
        sim_options = fsim_options
        model_struct_arrays = fmodel_struct_arrays
        pass  # data is already loaded
    else:
        print("No file with matching parameters and options found. Simulating data...")
        (
            equ_output,
            df,
            params,
            options,
            model_struct_arrays,
        ) = solve_and_simulate_data_jax(params, options, model_funcs, sim_options)

        # Save the data
        print("Simulation done. Dumping data now..")
        with open(datadir + "/data.pkl", "wb") as f:
            pickle.dump(
                (equ_output, df, params, options, sim_options, model_struct_arrays), f
            )

    return equ_output, df, params, options, sim_options, model_struct_arrays


def solve_and_simulate_data(params, options, sim_options):
    params, options = update_params_and_options(params=params, options=options)

    model_struct_arrays = create_model_struct_arrays(options=options)

    # Solve the model
    equ_output = jax.jit(
        lambda params_dict: equilibrium_solver(
            params=params_dict,
            options=options,
            model_struct_arrays=model_struct_arrays,
        )
    )(
        params,
    )

    # Simulate data
    df = simulate_data(
        equ_output=equ_output,
        options=options,
        params=params,
        sim_options=sim_options,
        state_space_arrays=model_struct_arrays,
    )

    return equ_output, df, params, options, model_struct_arrays


def solve_and_simulate_data_jax(params, options, model_funcs, sim_options):
    # seems redundant?
    params, options=model_funcs['update_params_and_options'](params=params, options=options)
    model_struct_arrays = create_model_struct_arrays(options=options, model_funcs=model_funcs)


    # Solve the model
    equ_output = jax.jit(
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
        equ_output=equ_output,
        options=options,
        params=params,
        sim_options=sim_options,
        state_space_arrays=model_struct_arrays,
    )

    return equ_output, df, params, options, model_struct_arrays


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


def simulate_chunk_jax(
    equ_output,
    options,
    sim_options,
    sim_options_chunk,
    state_space_arrays,
    seed,
    indexer,
    consumer_idxs,
    state_idxs,
    decision_idxs,
    ncells,
):
    partial_sim = lambda seed: simulate_with_solution_jax(
        model_solution=equ_output,
        options=options,
        sim_options=sim_options_chunk,
        state_space_arrays=state_space_arrays,
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


# @profile
def simulate_data_jax(equ_output, state_space_arrays, params, options, sim_options):
    # Does sampling runs to reach the number of requested observations,
    # by chunking the sampling runs into smaller chunks of chunk_size.
    # chunk_size has to be divisible by the number of requested observations

    num_agents = sim_options["num_agents"]
    chunk_size = sim_options["chunk_size"]

    n_full_chunks, last_chunk_size = divmod(num_agents, chunk_size)
    if last_chunk_size > 0:
        raise Exception(
            "the number of agents times observations has to be divisible by chunk_size"
        )

    nseeds = n_full_chunks
    seeds = np.arange(sim_options["seed"], nseeds + sim_options["seed"])

    # jax.jit(simulate_with_solution,) # This would be the way to jit it but I
    # would not work with because sim_options_chunk is one argument.
    sim_options_chunk = {
        "num_agents": chunk_size,
        "num_periods": sim_options["num_periods"],
    }

    dfs = []

    # Creating unique index for

    # I worry this indexer will scale poorly when consumer types and car types increase.
    # I want to use jit to speed up here
    # TODO: Create_data_frame needs to be vectorized.
    # Replace with some vector.

    # unpack options
    num_consumer_types = options["num_consumer_types"]
    num_car_types = options["num_car_types"]
    max_age_of_car_types = jnp.array(options["max_age_of_car_types"])
    num_states = state_space_arrays["state_space"].shape[0]
    num_decisions = state_space_arrays["decision_space"].shape[0]
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
        equ_output=equ_output,
        options=options,
        sim_options=sim_options,
        sim_options_chunk=sim_options_chunk,
        state_space_arrays=state_space_arrays,
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
    # dfs = map_simulate(seeds)

    # array approach
    # dfs_array = jnp.array([jit_simulate_chunk_jax(seed) for seed in seeds])
    # scan_simulate = jax.lax.scan(jit_simulate_chunk_jax, xs= seeds)
    # dfs = scan_simulate(seeds)

    # for seed in tqdm(seeds):
    #    sim_data = jit_simulate_chunk_jax(seed)
    #    dfs.append(sim_data)

    # vmap_simulate = jax.jit(jax.vmap(jit_simulate_chunk_jax))
    # dfs = vmap_simulate(seeds)

    #    for i in tqdm(range(n_full_chunks)):
    #        sim_dict = jit_sim(seed=seeds[i])
    #        sim_index=jnp.array([sim_dict['consumer_type'], sim_dict['state_idx'], sim_dict['decision_idx']])
    #
    #        # Columns: chunk_i, consumer_type, state_idx, decision_idx, counts
    #        chunk_is = jnp.repeat(i, ncells)
    #        # alternative could be jnp bincount across indexer
    #        #counts=jop.segment_sum(jnp.ones(chunk_size, dtype='int'), indexer[sim_index[0], sim_index[1], sim_index[2]])
    #        counts= jnp.bincount(indexer[sim_index[0], sim_index[1], sim_index[2]], minlength=ncells)
    #        sim_data=jnp.column_stack((chunk_is, consumer_idxs, state_idxs, decision_idxs, counts))
    #
    #        dfs.append(sim_data)

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
    # df = df.set_index(["chunk_i", "consumer_type", "state_idx", "decision_idx"])

    return df


# @profile
def simulate_data(equ_output, state_space_arrays, params, options, sim_options):
    # Does sampling runs to reach the number of requested observations,
    # by chunking the sampling runs into smaller chunks of chunk_size.
    # chunk_size has to be divisible by the number of requested observations

    num_agents = sim_options["num_agents"]
    chunk_size = sim_options["chunk_size"]

    n_full_chunks, last_chunk_size = divmod(num_agents, chunk_size)
    if last_chunk_size > 0:
        raise Exception(
            "the number of agents times observations has to be divisible by chunk_size"
        )

    nseeds = n_full_chunks
    seeds = np.arange(sim_options["seed"], nseeds + sim_options["seed"])

    # jax.jit(simulate_with_solution,) # This would be the way to jit it but I
    # would not work with because sim_options_chunk is one argument.
    sim_options_chunk = {
        "num_agents": chunk_size,
        "num_periods": sim_options["num_periods"],
    }

    partial_sim = lambda seed: simulate_with_solution_jax(
        model_solution=equ_output,
        options=options,
        sim_options=sim_options_chunk,
        state_space_arrays=state_space_arrays,
        seed=seed,
    )
    jit_sim = jax.jit(partial_sim)

    dfs = []
    for i in tqdm(range(n_full_chunks)):
        sim_dict = jit_sim(seed=seeds[i])
        sim_df = create_dataframe(sim_dict, sim_options_chunk, state_space_arrays)

        if sim_options["use_count_data"]:
            sim_df = obs_to_counts(sim_df, use_count_data=True)
        else:
            sim_df["counts"] = 1

        sim_df["chunk_i"] = i
        dfs.append(sim_df)

    # combining chunks into one df
    df = pd.concat(dfs, ignore_index=True)

    df = df.set_index(["chunk_i", "consumer_type", "state_idx", "decision_idx"])

    return df


def obs_to_counts(df, use_count_data=True):
    if use_count_data:
        df_count = (
            df.groupby(["consumer_type", "state_idx", "decision_idx"])
            .size()
            .reset_index(name="counts")
        )
        df = df_count.copy()
    else:
        df["counts"] = 1

    return df


def update_sim_index_to_est_index(index, sim_options):
    # This function takes the index of the simulated data and returns the index
    # of the data used for estimation. This is done by removing the chunk_i
    # index and replacing it with an updated index.
    chunk_size = sim_options["chunk_size"]
    estimation_size = sim_options["estimation_size"]
    num_agents = sim_options["num_agents"]

    assert np.all(
        index.names == ["chunk_i", "consumer_type", "state_idx", "decision_idx"]
    ), "The index names has been altered since simulation!"
    assert (
        estimation_size % chunk_size == 0
    ), "estimation_size should be a multiple of chunk_size"

    # dict asa map object from chunk_i to the new index
    chunk_index_values = index.get_level_values("chunk_i").unique()
    j = 0

    estimation_index = np.arange(num_agents // estimation_size)
    estimation_index = np.repeat(
        estimation_index, estimation_size // chunk_size
    ).astype(int)
    map_chunk_index_to_estimation_index = dict(
        zip(chunk_index_values, estimation_index)
    )

    # Creating new index
    df_idx = pd.DataFrame(index=index).reset_index()

    df_idx["est_i"] = (
        df_idx["chunk_i"].map(map_chunk_index_to_estimation_index).astype("Int32")
    )
    df_idx = df_idx.set_index(
        ["chunk_i", "est_i", "consumer_type", "state_idx", "decision_idx"]
    )

    return df_idx.index


# @profile
def calculate_cfps(df, state_decision_arrays, calc_counts=False):
    # makes a frequency estimator for ccps i.e conditional on states
    df = df.set_index(["consumer_type", "state_idx", "decision_idx"])
    denom = df.groupby(["consumer_type", "state_idx"]).counts.sum()
    num = df.groupby(["consumer_type", "state_idx", "decision_idx"]).counts.sum()
    if calc_counts:
        df_freq = num
    else:
        df_freq = num / denom
    df_freq = df_freq.to_frame(name="freq_ccp")

    # Constructing state dependent frequency estimator
    cfps = (
        np.zeros(
            (
                df.index.get_level_values("consumer_type").unique().shape[0],
                state_decision_arrays["state_space"].shape[0],
                state_decision_arrays["decision_space"].shape[0],
            )
        )
        + np.nan
    )
    for n in range(cfps.shape[0]):
        for s in range(cfps.shape[1]):
            for d in range(cfps.shape[2]):
                try:
                    cfps[n, s, d] = df_freq.loc[(n, s, d), "freq_ccp"]
                except KeyError:
                    continue

    return cfps


def calculate_cfps_from_df(df, calc_counts=False):
    df = df.set_index(["consumer_type", "state_idx", "decision_idx"])
    # denom = df.groupby(["consumer_type", "state_idx"]).counts.sum()
    denom = df.groupby(["consumer_type", "state_idx"]).counts.transform("sum")
    num = df.groupby(["consumer_type", "state_idx", "decision_idx"]).counts.sum()

    if calc_counts:
        df_freq = num
    else:
        df_freq = num / denom

    df_freq = df_freq.to_frame(name="ccps")

    df_freq.index = df_freq.index.rename(names=["tau", "state", "decision"])
    df_freq = df_freq.reset_index().set_index(["tau", "decision", "state"]).sort_index()
    # df_freq['counts'] = num

    if calc_counts:
        df_freq = df_freq.rename(columns={"ccps": "counts"})

    return df_freq


def true_ccps(equ_output):
    assert (
        equ_output["ccps_tau"].shape[0] == 1
    ), "This function only works for one consumer type"
    ccps = equ_output["ccps_tau"][0, :, :]

    return ccps


def calculate_scrap_probabilities(df):
    """
    returns: consumer_type x states Array scrappage probabilities at each state
    estimated from scrap_counts from df.
    """

    df = df.set_index(["consumer_type", "state_idx"])
    df = df.groupby(level=["consumer_type", "state_idx"]).sum()

    df["scrap_prob"] = df["scrap_counts"] / df["counts"]

    scrap_probabilities = df["scrap_prob"].unstack(level=1).values

    return scrap_probabilities


def estimate_clogit(
    df,
    state_decision_arrays,
    equ_output,
    w_j_vars,
    s_i_vars,
    w_j_deg,
    s_i_deg,
    unwanted_choices=None,
):
    # estimate_clogit.py
    # We will estimate conditional logit models with the following specification of utility:
    #     u_{ij}  = x_{ij} \beta + \epsilon_{ij}
    #             = w_{j} \beta_w  + z_{ij} \beta_z  + \epsilon_{ij}
    # where x_{ij} is a vector of K=K_w+K_z explanatory variables
    # - w_{j} is a k_w vector of explanatory variables that only vary by decision
    # - z_{ij} is a k_z vector of variables that only vary by state and decision and factors into
    # z_{ij} takes the form z_{ij} = w_{j} \otimes s_i
    # where \otimes is the kronecker product and s_{i} is a vector of individual specific variables
    # that only depends on the state of the individual.

    # Examples of w_{j} are
    # - age of the chosen car (and polynomials in age)
    # - dummy for purchasing a new car
    # - dummies for trading, purging or keeping the existing car
    # - or dummies for all chosen car types
    # Examples of z_{ij} are
    # - age of existing car interacted with dummy for trading
    # - age of existing car interacted with dummy for purging
    # - age of existing car interacted with dummy for keeping
    # - dummies for all states interacted with dummies for all decisions

    # %%
    # 2. Read in state and decision space, simulated data, and choice probabilities

    # Read in simulated data (passed into the function)
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]

    # Create state and decision indexes and useful tools
    state_idx = np.arange(state_space.shape[0])
    n_decisions = decision_space.shape[0]
    decision_idx = np.arange(n_decisions)

    # Parse data on observed choices and states
    y = df.decision_idx.values
    state_data = df.state_idx.values
    counts = df["counts"].values

    # %%
    # 3. Create explanatory variables x for conditional logit and estimate the model

    # x-array for *all* states and decisions
    x_space, name_space = create_x_matrix(
        state_space,
        decision_space,
        state_idx,
        w_j_vars,
        w_j_deg,
        s_i_vars,
        s_i_deg,
        unwanted_choices=unwanted_choices,
    )

    # x-array for *observed* states and decisions
    x, name = create_x_matrix(
        state_space,
        decision_space,
        state_data,
        w_j_vars,
        w_j_deg,
        s_i_vars,
        s_i_deg,
        unwanted_choices=unwanted_choices,
    )

    # %%

    # 3.1. Estimate the Logit model
    res = clogit(
        y,
        x,
        counts,
        cov_type="Ainv",
        theta0=np.zeros(x.shape[2]),
        deriv=2,
        parnames=name,
        quiet=False,
    )

    # predict smoothed ccps:
    ccps_logit_all_states = ccp(utility(res.theta_hat, x_space))
    index_df = pd.DataFrame(state_space, columns=["no_car", "age"])
    index_df.set_index(["no_car", "age"], inplace=True)
    est_ccps = pd.DataFrame(
        ccps_logit_all_states, index=index_df.index, columns=decision_idx
    )

    return est_ccps.values


def show_plots(ccps, ccps_logit_all_states, cfps, state_decision_arrays, save_to=None):
    # unpack state and decision space
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]

    # Create state and decision indexes and useful tools
    state_idx = np.arange(state_space.shape[0])
    _, n_ages = np.unique(
        decision_space[decision_space[:, 1] > 0, 1], return_counts=True
    )
    n_decisions = decision_space.shape[0]
    decision_idx = np.arange(n_decisions)

    # %%
    # Plot prepping:

    # indexes for decisions (TODO should be in options)
    id_keep = 0  # first row in decision space
    id_purge = -1  # last row in decision space

    # indexes for new cars
    new_cars_bool = (decision_space[:, 0] == 2.0) & (decision_space[:, 2] == 0.0)
    d_indexer = np.column_stack((decision_space, np.arange(n_decisions)))
    id_news = d_indexer[
        new_cars_bool, 3
    ]  # Can be a vector if more than one new car type

    # Constructs the correct ordering ie. new car comes first.
    # Creates a list of indices for the cars to plot
    id_cars = [
        [id_new] + list(range(1, n_ages[i]) + (id_new - n_ages[i]))
        for i, id_new in enumerate(id_news)
    ]
    id_cars = [item for sublist in id_cars for item in sublist]
    id_plot = np.array(id_cars + [id_keep] + [id_purge], dtype=int)

    # %%
    # Plot 0:

    # %%
    # 3.4 Plot aggregate choice probabilities

    # compare the predicted "Market shares" based on logit ccps, true ccps, and observed frequencies
    # true ccps aggregated over observed states
    s_true = np.mean(ccps, axis=0)

    # estimated logit ccps aggregated over observed states
    s_logit = np.mean(ccps_logit_all_states, axis=0)

    # When expanding to multiple consumer types use: ['consumer_type', 'state_idx', 'decision_idx']
    df_freq = pd.DataFrame(cfps, index=state_idx, columns=decision_idx)
    df_freq = df_freq.stack()
    df_freq.index = df_freq.index.set_names(
        ["state_idx", "decision_idx"], inplace=False
    )
    df_freq = df_freq.to_frame(name="freq_ccp")

    # Next step is to make a simple average across states.
    df_freq["weights"] = 1 / n_decisions
    # If you are in clunker state add - 1 in the denominator of the weights
    clunker_bool = np.array(
        [
            (state_space[:, 0] == i + 1) & (state_space[:, 1] == n_ages[i])
            for i in range(n_ages.shape[0])
        ]
    ).sum(axis=0, dtype=bool)
    clunkers_idx = state_idx[clunker_bool]
    clunker_rows = pd.IndexSlice[clunkers_idx, :]
    df_freq.loc[clunker_rows, "weights"] = 1 / (n_decisions - 1)
    # clunker state and keep decision are illegal and weight is set to zero
    # clunkerillegal = pd.IndexSlice[clunkers_idx,0]
    # df_freq.loc[clunkerillegal, 'weights'] = 0

    # If you are in no car state add - 1 in the denominator of the weights
    no_car_idx = state_idx[state_space[:, 0] == 0]
    # no car state and keep decision are illegal and weight is set to zero
    nocar_rows = pd.IndexSlice[no_car_idx, :]
    df_freq.loc[nocar_rows, "weights"] = 1 / (n_decisions - 1)
    # nocarillegal = pd.IndexSlice[no_car_idx,0]
    # df_freq.loc[nocarillegal, 'weights'] = 0

    # assert np.all(df_freq.groupby('state_idx')['weights'].sum() == 1.0)

    # construct weighted frequency ccp
    df_freq["w_freq_ccp"] = df_freq["freq_ccp"] * df_freq["weights"]

    # aggregate over states
    s_freq_lookup = df_freq.groupby(["decision_idx"]).w_freq_ccp.sum()

    # In case of any missing decisions in the data, we need to fill in zeros
    s_freq = np.zeros((n_decisions))
    for j in range(n_decisions):
        try:
            s_freq[j] = s_freq_lookup[j]
        except KeyError:
            s_freq[j] = 0.0

    # plot the results
    plt.plot(s_true[id_plot], label="true")
    plt.plot(s_logit[id_plot], label="logit")
    plt.plot(s_freq[id_plot], label="freq")
    plt.title("Aggregate Choice Probabilities")
    plt.xlabel("Decision Variable")
    plt.legend()
    if save_to is not None:
        plt.savefig(save_to + "Aggregate_Choice_Probabilities.png")

    # %%
    # Plot 1: 3d true ccp estimator
    # Create a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xvar, yvar = np.meshgrid(np.arange(ccps.shape[0]), np.arange(ccps.shape[1]))

    # Plot the surface
    surf = ax.plot_surface(xvar, yvar, ccps[:, id_plot].T, cmap="viridis")

    # Add labels and title
    ax.set_xlabel("State Variable")
    ax.set_ylabel("Decision Variable")
    ax.set_zlabel("CCPs")
    ax.set_title("3D Surface Plot of CCPs")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # %%
    # Plot 2: 3d frequency estimator

    # Create a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xvar, yvar = np.meshgrid(np.arange(ccps.shape[0]), np.arange(ccps.shape[1]))

    # Plot the surface
    surf = ax.plot_surface(xvar, yvar, cfps[:, id_plot].T, cmap="viridis")

    # Add labels and title
    ax.set_xlabel("State Variable")
    ax.set_ylabel("Decision Variable")
    ax.set_zlabel("CFPs")
    ax.set_title("3D Surface Plot of CFPs")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_to is not None:
        plt.savefig(save_to + "3D_Surface_Plot_of_CFPs.png")

    # %%
    # Plot 3:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})

    # Subplot 1: ccps
    xvar, yvar = np.meshgrid(np.arange(ccps.shape[0]), np.arange(ccps.shape[1]))
    surf1 = axes[0].plot_surface(xvar, yvar, ccps[:, id_plot].T, cmap="viridis")
    axes[0].set_title("CCPs")
    axes[0].set_xlabel("State Variable")
    axes[0].set_ylabel("Decision Variable")
    axes[0].set_zlabel("CCPs")
    fig.colorbar(surf1, ax=axes[0], shrink=0.5, aspect=5)

    # Subplot 2: ccps_logit_all_states
    surf2 = axes[1].plot_surface(
        xvar, yvar, ccps_logit_all_states[:, id_plot].T, cmap="viridis"
    )
    axes[1].set_title("CCPs Logit All States")
    axes[1].set_xlabel("State Variable")
    axes[1].set_ylabel("Decision Variable")
    axes[1].set_zlabel("CCPs Logit")
    fig.colorbar(surf2, ax=axes[1], shrink=0.5, aspect=5)

    # Subplot 3: Differences
    diffs = ccps_logit_all_states[:, id_plot] - ccps[:, id_plot]
    surf3 = axes[2].plot_surface(xvar, yvar, diffs.T, cmap="viridis")
    axes[2].set_title("Differences (CCPs Logit - CCPs)")
    axes[2].set_xlabel("State Variable")
    axes[2].set_ylabel("Decision Variable")
    axes[2].set_zlabel("Differences")
    fig.colorbar(surf3, ax=axes[2], shrink=0.5, aspect=5)

    # %%
    # Show plots
    if save_to is not None:
        plt.savefig(save_to + "true_ccp_differences.png")

    plt.show()


def DDR_regression(
    ccps,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    linear_specification,
):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """

    X, model_specification = construct_regvars_for_all_taus(
        ccps,
        prices,
        scrap_probabilities,
        state_decision_arrays,
        params,
        options,
        linear_specification,
    )

    rows_dropped = X["ccp"].isna().sum()
    print("dropping {} rows with ccps=0.0 ".format(rows_dropped))
    X = X.dropna(subset=["ccp"])

    # assert np.all(np.isclose(X.groupby(level=1)['ccp'].sum(), 1.0)), "You have ccps that do not sum to 1!!!"

    Y, Xreg = utils.create_regvars(X, model_specification)

    B = np.linalg.lstsq(Xreg, Y, rcond=None)[0]

    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def create_counts_in_tab_format(counts, state_decision_arrays, params, options):
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])
    nconsumers = options["num_consumer_types"]

    # prepping data for create_tab function
    index_array = np.vstack(
        [
            np.hstack(
                [
                    np.repeat(tau, repeats=state_space.shape[0]).reshape(-1, 1),
                    state_space,
                ]
            )
            for tau in range(nconsumers)
        ]
    )
    index_df = pd.DataFrame(index_array, columns=["tau", "no_car", "age"])
    index_df.set_index(["tau", "no_car", "age"], inplace=True)

    # restacking ccps to 2d array
    counts = np.vstack([counts[tau, ...] for tau in range(nconsumers)])

    # DataFraming counts
    counts_tab = pd.DataFrame(counts, index=index_df.index, columns=decision_idx)

    # add ev column - (does not do anything)
    counts_tab["ev"] = np.nan

    tabs = []
    for tau in range(0, nconsumers):
        # prepare data for regression
        tab_tau = utils.create_tab(
            counts_tab.loc[tau, :], state_decision_arrays=state_decision_arrays
        )
        tab_tau["tau"] = tau
        tabs.append(tab_tau)
    tab = pd.concat(tabs, axis=0)
    tab.rename(columns={"ccp": "counts"}, inplace=True)

    # remove all infeasible actions from tab
    I_feasible = mi.feasible_choice_all(
        tab["state"].values,
        tab["decision"].values,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    tab = tab.loc[np.array(I_feasible), :]

    # set indices
    tab = tab.set_index(["tau", "state", "decision"])

    return tab


# @profile
def wls_regression_mc(ccps, X, counts, model_specification):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # Index for zero share rows
    # I = ccps.values.flatten() != 0.0

    X = X[model_specification]
    X = X.values.astype(float)

    # X = X[I, :]

    # ccps = ccps.loc[I, :]
    logY = np.log(ccps.values.flatten())

    # counts = counts.loc[I, :]

    B = wls.estimate_wls(logY, X, counts)

    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def wls_regression(
    ccps, counts, prices, state_decision_arrays, params, options, linear_specification
):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    X, model_specification = construct_regvars_for_all_taus(
        ccps, prices, state_decision_arrays, params, options, linear_specification
    )

    # calculating a tab version of counts (same order as X)
    counts_tab = create_counts_in_tab_format(
        counts, state_decision_arrays, params, options
    )

    # creating dependent variable and regressors
    logY, Xreg = utils.create_regvars(
        X,
        model_specification,
    )

    B = wls.estimate_wls(logY, Xreg, counts_tab)

    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def construct_tab_index_for_all_taus(
    ccps, prices, state_decision_arrays, params, options, linear_specification
):
    # unpack state and decision space
    # TODO: This could be done outside the function potentially leading to speed ups.
    # TODO: I think there is something here that could be done in genaral and then adapted to the incoming data.
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])
    nconsumers = options["num_consumer_types"]

    index_array = np.vstack(
        [
            np.hstack(
                [
                    np.repeat(tau, repeats=state_space.shape[0]).reshape(-1, 1),
                    state_space,
                ]
            )
            for tau in range(nconsumers)
        ]
    )
    # prepping data for create_tab function
    index_df = pd.DataFrame(index_array, columns=["tau", "no_car", "age"])
    index_df.set_index(["tau", "no_car", "age"], inplace=True)

    # a loop version would look like this:

    # Iota is independent of consumer type so we can reuse it.
    iota = utils.create_iota_space(
        state_decision_arrays=state_decision_arrays, params=params, options=options
    )
    Xs = []
    model_specification = []

    for tau in range(0, nconsumers):
        ccps_tau = ccps.loc[tau, :]

        # prepare data for regression
        tab_tau = utils.create_tab(
            ccps_tau, state_decision_arrays=state_decision_arrays
        )


def construct_regvars_for_all_taus(
    ccps,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    linear_specification,
):
    # unpack state and decision space
    # TODO: This could be done outside the function potentially leading to speed ups.
    # TODO: I think there is something here that could be done in genaral and then adapted to the incoming data.
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])
    nconsumers = options["num_consumer_types"]

    index_array = np.vstack(
        [
            np.hstack(
                [
                    np.repeat(tau, repeats=state_space.shape[0]).reshape(-1, 1),
                    state_space,
                ]
            )
            for tau in range(nconsumers)
        ]
    )
    # prepping data for create_tab function
    index_df = pd.DataFrame(index_array, columns=["tau", "no_car", "age"])
    index_df.set_index(["tau", "no_car", "age"], inplace=True)

    # restacking ccps to 2d array
    ccps = np.vstack([ccps[tau, ...] for tau in range(nconsumers)])

    # DataFraming ccps
    ccps = pd.DataFrame(ccps, index=index_df.index, columns=decision_idx)

    # add ev column - (does not do anything)
    ccps["ev"] = np.nan

    # a loop version would look like this:

    # Iota is independent of consumer type so we can reuse it.
    iota = utils.create_iota_space(
        state_decision_arrays=state_decision_arrays, params=params, options=options
    )
    Xs = []
    model_specification = []

    for tau in range(0, nconsumers):
        ccps_tau = ccps.loc[tau, :]

        # prepare data for regression
        tab_tau = utils.create_tab(
            ccps_tau, state_decision_arrays=state_decision_arrays
        )

        X_tau, model_specification_tau = utils.create_X_matrix_from_tab(
            tab_tau,
            iota,
            prices,
            scrap_probabilities,
            state_decision_arrays,
            params,
            options,
            tau=tau,
            linear_specification=linear_specification,
        )
        # renaming variables to indicate that they are consumer type specific
        model_specification_tau_newname = [
            s + f"_{tau}" for s in model_specification_tau
        ]
        cols_rename_tau = dict(
            zip(model_specification_tau, model_specification_tau_newname)
        )
        X_tau = X_tau.rename(columns=cols_rename_tau)

        # resetting index to add consumer type
        X_tau["tau"] = tau
        X_tau = X_tau.reset_index().set_index(["tau", "decision", "state"])

        Xs.append(X_tau)
        # models.append(model_specification_tau_newname)
        model_specification = model_specification + model_specification_tau_newname

    X = pd.concat(Xs, axis=0).fillna(0.0)

    return X, model_specification


def calculate_accident_parameters(scrap_probabilities):
    """This function calculates the parameters of the accident model."""

    scrap_probabilities_new_car = scrap_probabilities[:, 0]

    zero_probs = scrap_probabilities_new_car == 0.0

    logit_inv = lambda p: jnp.log(p / (1 - p))

    acc_0 = logit_inv(scrap_probabilities_new_car)

    acc_0 = acc_0.at[zero_probs].set(-100)  # Jax lingo for setting values

    return NotImplementedError(
        "Scrap probabilities from simulated data do not include accidents so this approach is not functional, hence this error."
    )  # acc_0


def PDR_regression(
    ccps, prices, state_decision_arrays, params, options, linear_specification=False
):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with consumer types
    and state_decision_arrays

    """
    # returns X and model_specification
    X, model_specification = construct_regvars_for_all_taus(
        ccps,
        prices,
        state_decision_arrays,
        params,
        options,
        linear_specification,
    )

    # Creating a version of data specific to the PDR regression.
    Xpdr = X.copy().fillna(0.0)
    # This is only done to get starting values.
    X = X.dropna(subset=["ccp"])

    X = X[~(X["ccp"] == 0.0)]

    # For starting values
    logY, Xreg = utils.create_regvars(X, model_specification)
    # for PDR regression
    _, Xpdrreg = utils.create_regvars(Xpdr, model_specification)
    Y = Xpdr["ccp"].values

    # pdr regression.
    # starting values
    b0 = np.linalg.lstsq(Xreg, logY, rcond=None)[0]

    # Here I would like to call a procedure that does the pdr regression.
    B, iter, diff, ests = pdr.estimate_pdr(Y, Xpdrreg, b0)

    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def pdr_regression_mc(ccps, X, model_specification):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # Index for zero share rows
    I = ccps.values.flatten() != 0.0

    X = X[model_specification]
    X = X.values.astype(float)

    # X = X[I, :]

    # ccps = ccps.loc[I, :]
    logY = np.log(ccps.values.flatten())

    # counts = counts.loc[I, :]

    # compute starting values
    g0 = np.linalg.lstsq(X[I], logY[I], rcond=None)[0]

    B, iter, diff, _ = pdr.estimate_pdr(ccps.values.flatten(), X, g0)

    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def PDR_regression_experimental(ccps, prices, state_decision_arrays, params, options):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # unpack state and decision space
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])

    # prepping data for create_tab function
    index_df = pd.DataFrame(state_space, columns=["no_car", "age"])
    index_df.set_index(["no_car", "age"], inplace=True)
    ccps = pd.DataFrame(ccps, index=index_df.index, columns=decision_idx)

    # add ev column - (does not do anything)
    ccps["ev"] = np.nan

    # prepare data for regression
    tab = utils.create_tab(ccps, state_decision_arrays=state_decision_arrays)
    iota = utils.create_iota_space(
        state_decision_arrays=state_decision_arrays, params=params, options=options
    )

    tau = 1  # just a placeholder for now
    X = utils.create_X_matrix_from_tab_experimental(
        tab, iota, prices, state_decision_arrays, params, options, tau=1
    )

    # Creating a version of data specific to the PDR regression.
    Xpdr = X.copy().fillna(0.0)
    # This is only done to get starting values.
    X = X.dropna(subset=["ccp"])

    # For starting values
    Xreg = X[X.columns[1:]].values.astype(float)
    logY = np.log(X["ccp"].values)
    # logY , Xreg, param_names = utils.create_regvars_for_tau(X, state_decision_arrays, tau=tau)
    # for PDR regression
    Xpdrreg = Xpdr[Xpdr.columns[1:]].values.astype(float)
    # _ , Xpdrreg, param_names = utils.create_regvars_for_tau(Xpdr, state_decision_arrays, tau=tau)
    Y = Xpdr["ccp"].values

    # pdr regression.
    # starting values
    b0 = np.linalg.lstsq(Xreg, logY, rcond=None)[0]

    # Here I would like to call a procedure that does the pdr regression.
    B, iter, diff, ests = pdr.estimate_pdr(Y, Xpdrreg, b0, max_iter=100, tol=1e-6)
    est = pd.DataFrame(B, index=[X.columns[1:]], columns=["Coefficient" + str(tau)])
    return est


def extract_true_structural_parameters(
    equ_output, state_decision_arrays, params, options
):
    # unpack state and decision space
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])

    # parameters

    # mum
    if options["num_consumer_types"] > 1:
        mum = extract_from_params(
            params, "mum", "price_all_{:d}", varies_with="consumer_type"
        )
    elif options["num_consumer_types"] == 1:
        mum = extract_from_params(params, "mum", "price_all_all", varies_with="scalar")

    # scrap_correction
    scrap_correction = extract_from_params(
        params, "sigma_sell_scrapp", "scrap_correction_all_all", varies_with="scalar"
    )

    # ptranscost
    if options["num_consumer_types"] > 1:
        psych_transcost = extract_from_params(
            params, "psych_transcost", "buying_all_{:d}", varies_with="consumer_type"
        )
    elif options["num_consumer_types"] == 1:
        psych_transcost = extract_from_params(
            params, "psych_transcost", "buying_all_all", varies_with="scalar"
        )

    # u_0
    if options["num_consumer_types"] > 1:
        u_0 = extract_from_params(
            params, "u_0", "car_type_{:d}_{:d}", varies_with="both"
        )
    elif options["num_consumer_types"] == 1:
        u_0 = extract_from_params(
            params, "u_0", "car_type_{:d}_all", varies_with="car_type"
        )

    # u_a
    if options["num_consumer_types"] > 1:
        u_a = extract_from_params(
            params, "u_a", "car_type_{:d}_x_age_{:d}", varies_with="both"
        )
    elif options["num_consumer_types"] == 1:
        u_a = extract_from_params(
            params, "u_a", "car_type_{:d}_x_age_all", varies_with="car_type"
        )

    # u_a_sq
    if options["num_consumer_types"] > 1:
        u_a_sq = extract_from_params(
            params, "u_a_sq", "car_type_{:d}_x_age_sq_{:d}", varies_with="both"
        )
    elif options["num_consumer_types"] == 1:
        u_a_sq = extract_from_params(
            params, "u_a_sq", "car_type_{:d}_x_age_sq_all", varies_with="car_type"
        )

    # u_even
    if options["num_consumer_types"] > 1:
        u_even = extract_from_params(
            params, "u_even", "car_type_{:d}_x_age_even_{:d}", varies_with="both"
        )
    elif options["num_consumer_types"] == 1:
        u_even = extract_from_params(
            params, "u_even", "car_type_{:d}_x_age_even_all", varies_with="car_type"
        )

    # Ev terms
    ev_terms = equ_output["ev_tau"]

    ev_terms = [(i, j, value) for (i, j), value in np.ndenumerate(ev_terms)]
    ev_terms = pd.DataFrame(
        ev_terms, columns=["consumer_type", "state_idx", "true values"]
    )

    ev_terms["consumer_type"] = ev_terms["consumer_type"].astype(int)
    ev_terms["s_type"] = state_space[ev_terms["state_idx"], 0].astype(int)
    ev_terms["s_age"] = state_space[ev_terms["state_idx"], 1].astype(int)
    s_type_vec = ev_terms["s_type"].values
    s_age_vec = ev_terms["s_age"].values
    consumer_type_vec = ev_terms["consumer_type"].values
    ev_terms["variablename"] = [
        f"ev_dums_{s_type}_{s_age}_{consumer_type}"
        for s_type, s_age, consumer_type in zip(
            s_type_vec, s_age_vec, consumer_type_vec
        )
    ]

    ev_terms = ev_terms[["variablename", "true values"]].set_index("variablename")

    df = pd.concat(
        [mum, scrap_correction, psych_transcost, u_0, u_a, u_a_sq, u_even, ev_terms],
        axis=0,
    )

    df = df[["true values"]]

    return df


def extract_from_params(params, key, string_format, varies_with: str):
    """'varies_with' takes the options:

    - 'scalar'
    - 'consumer_type'
    - 'car_type'
    - 'both'

    """

    # Cannot figure this out atm
    values = params[key]
    if varies_with == "consumer_type":
        values = [(tau[0], value) for tau, value in np.ndenumerate(values)]
        df = pd.DataFrame(values, columns=["consumer_type", "true values"])
        consumer_type_vec = df["consumer_type"].values
        df["variablename"] = [string_format.format(tau) for tau in consumer_type_vec]

    elif varies_with == "car_type":
        values = [
            (car_type[1] + 1, value) for car_type, value in np.ndenumerate(values)
        ]
        df = pd.DataFrame(values, columns=["car_type", "true values"])
        car_type_vec = df["car_type"].values
        df["variablename"] = [string_format.format(tau) for tau in car_type_vec]

    elif varies_with == "both":
        values = [
            (tau, car_type + 1, value)
            for (tau, car_type), value in np.ndenumerate(values)
        ]
        df = pd.DataFrame(values, columns=["consumer_type", "car_type", "true values"])
        consumer_type_vec = df["consumer_type"].values
        car_type_vec = df["car_type"].values
        df["variablename"] = [
            string_format.format(car_type, tau)
            for tau, car_type in zip(consumer_type_vec, car_type_vec)
        ]

    elif varies_with == "scalar":
        df = pd.DataFrame([values], columns=["true values"])
        df["variablename"] = string_format

    return df.set_index("variablename")
