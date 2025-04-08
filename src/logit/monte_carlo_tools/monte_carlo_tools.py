import numpy as np
import pandas as pd
import jax.numpy as jnp

def update_sim_index_to_est_index(index, sim_options):
    # This function takes the index of the simulated data and returns the index
    # of the data used for estimation. This is done by removing the chunk_i
    # index and replacing it with an updated index.
    chunk_size = sim_options["chunk_size"]
    estimation_size = sim_options["estimation_size"]
    n_agents = sim_options["n_agents"]

    assert np.all(
        index.names == ["chunk_i", "consumer_type", "state", "decision"]
    ), "The index names has been altered since simulation!"
    assert (
        estimation_size % chunk_size == 0
    ), "estimation_size should be a multiple of chunk_size"

    # dict asa map object from chunk_i to the new index
    chunk_index_values = index.get_level_values("chunk_i").unique()

    estimation_index = np.arange(n_agents // estimation_size)
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
        ["chunk_i", "est_i", "consumer_type", "state", "decision"]
    )

    return df_idx.index

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


def extract_true_structural_parameters(
    model_solution, model_struct_arrays, params, options
):
    # unpack state and decision space
    decision_space = model_struct_arrays["decision_space"]
    state_space = model_struct_arrays["state_space"]
    state_idx = np.arange(state_space.shape[0])
    decision_idx = np.arange(decision_space.shape[0])

    # parameters

    # mum
    if options["n_consumer_types"] > 1:
        mum = extract_from_params(
            params, "mum", "price_all_{:d}", varies_with="consumer_type"
        )
    elif options["n_consumer_types"] == 1:
        mum = extract_from_params(params, "mum", "price_all_all", varies_with="scalar")

    # scrap_correction
    scrap_correction = extract_from_params(
        params, "sigma_sell_scrapp", "scrap_correction_all_all", varies_with="scalar"
    )

    # ptranscost
    if options["n_consumer_types"] > 1:
        psych_transcost = extract_from_params(
            params, "psych_transcost", "buying_all_{:d}", varies_with="consumer_type"
        )
    elif options["n_consumer_types"] == 1:
        psych_transcost = extract_from_params(
            params, "psych_transcost", "buying_all_all", varies_with="scalar"
        )

    # u_0
    if options["n_consumer_types"] > 1:
        u_0 = extract_from_params(
            params, "u_0", "car_type_{:d}_{:d}", varies_with="both"
        )
    elif options["n_consumer_types"] == 1:
        u_0 = extract_from_params(
            params, "u_0", "car_type_{:d}_all", varies_with="car_type"
        )

    # u_a
    if options["n_consumer_types"] > 1:
        u_a = extract_from_params(
            params, "u_a", "car_type_{:d}_x_age_{:d}", varies_with="both"
        )
    elif options["n_consumer_types"] == 1:
        u_a = extract_from_params(
            params, "u_a", "car_type_{:d}_x_age_all", varies_with="car_type"
        )

    # u_a_sq
    if options["n_consumer_types"] > 1:
        u_a_sq = extract_from_params(
            params, "u_a_sq", "car_type_{:d}_x_age_sq_{:d}", varies_with="both"
        )
    elif options["n_consumer_types"] == 1:
        u_a_sq = extract_from_params(
            params, "u_a_sq", "car_type_{:d}_x_age_sq_all", varies_with="car_type"
        )

    # u_even
    if options["n_consumer_types"] > 1:
        u_even = extract_from_params(
            params, "u_even", "car_type_{:d}_x_age_even_{:d}", varies_with="both"
        )
    elif options["n_consumer_types"] == 1:
        u_even = extract_from_params(
            params, "u_even", "car_type_{:d}_x_age_even_all", varies_with="car_type"
        )

    # Ev terms
    ev_terms = model_solution["ev_tau"]

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