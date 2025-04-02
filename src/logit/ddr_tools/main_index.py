import numpy as np
import pandas as pd
def create_main_feasible_idx(model_struct_arrays, params, options):

    tab_idx = create_tab_index(
        model_struct_arrays=model_struct_arrays,
        options=options,
    )

    feasible_states = create_feasible_choice_indexer(
    tab_idx.get_level_values("state").values,
    tab_idx.get_level_values("decision").values,
    model_struct_arrays=model_struct_arrays,
    params=params,
    options=options,
    )
    return tab_idx[feasible_states]

def create_tab_index(model_struct_arrays, options):
    decision_space = model_struct_arrays["decision_space"]
    state_space = model_struct_arrays["state_space"]
    n_consumer_types = options["n_consumer_types"]

    nD = model_struct_arrays["decision_space"].shape[0]  # array of decision space
    nS = model_struct_arrays["state_space"].shape[0]  # array of state space

    index_array = np.vstack(
        [
            np.hstack(
                [
                    np.repeat(tau, repeats=state_space.shape[0]).reshape(-1, 1),
                    state_space,
                ]
            )
            for tau in range(n_consumer_types)
        ]
    )
    index_df = pd.DataFrame(index_array, columns=["tau", "no_car", "age"])
    index_df.set_index(["tau", "no_car", "age"], inplace=True)

    # verify ordering and construct ss index
    current_consumer_type = 0
    j = 0
    ss = np.zeros((nS * n_consumer_types)) + np.nan
    sss = np.arange(nS)
    for i, (c_type, s_type, s_age) in enumerate(index_df.index.values):
        if c_type != current_consumer_type:
            current_consumer_type = c_type
            j = 0
        s = model_struct_arrays["state_space"][j, :]
        ss[i] = sss[j]
        assert (s[0] == s_type) & (s[1] == s_age)
        j += 1

    dd = np.arange(nD)
    consumertypes = np.repeat(
        np.array(index_df.index.get_level_values(0).tolist()), nD
    ).reshape(-1, nD)
    tab = pd.DataFrame(consumertypes, index=ss, columns=dd)
    tab.index.name = "state"
    tab.columns.name = "decision"

    # rename variables:
    tab = tab.reset_index().melt(id_vars="state", value_name="tau")
    tab[["decision", "state"]] = tab[["decision", "state"]].astype(
        int
    )  # not sure why this gets converted to 'O'...

    tab = tab.set_index(["tau", "decision", "state"]).sort_index()
    
    return tab.index


def create_feasible_choice_indexer(sidx_vec, didx_vec, model_struct_arrays, params, options):
    N = sidx_vec.size
    assert (
        didx_vec.size == N
    ), f"sidx_vec and didx_vec should have the same length but have {sidx_vec.size} and {didx_vec.size}"
    state_space = np.array(model_struct_arrays["state_space"])
    decision_space = np.array(model_struct_arrays["decision_space"])
    nS = state_space.shape[0]
    nD = decision_space.shape[0]

    # only two decisions are illegal
    I_nocar = state_space[sidx_vec, 0] == 0
    I_purge = decision_space[didx_vec, 0] == 1
    I_keep = decision_space[didx_vec, 0] == 0
    # I_has_clunker = state_space[sidx_vec, 1] == options["max_age_of_car_types"][state_space[sidx_vec, 0]-1]
    assert (
        np.unique(options["max_age_of_car_types"]).size == 1
    ), "All car types must have the same maximum age for this code to work"
    abar = options["max_age_of_car_types"][0]
    I_has_clunker = state_space[sidx_vec, 1] == abar

    I_illegal1 = I_nocar & I_keep
    I_illegal2 = I_has_clunker & I_keep
    I_feasible = (~I_illegal1) & (~I_illegal2)

    return I_feasible