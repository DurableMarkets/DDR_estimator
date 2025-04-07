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

