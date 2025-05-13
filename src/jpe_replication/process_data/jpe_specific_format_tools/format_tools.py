import numpy as np

def translate_state_indices(df_states):
    """
    df_states: a df containing the state indices present in the data
    """
    # doing some assertion checks
    assert (
        df_states.columns == ["s_car_type", "s_car_age"]
    ).all(), "df_states must have columns s_car_type and s_car_age"
    assert df_states["s_car_type"].min() >= -1, "s_car_type must be >= -1"
    assert df_states["s_car_age"].min() >= -1, "s_car_age must be >= -1"
    assert np.any(df_states["s_car_age"] != 0), "s_car_age cannot be 0"
    assert df_states["s_car_age"].max() <= 25, "s_car_age must be <= 25"

    ## Translate indices
    cols_s = ["s_car_type", "s_car_age"]
    state_space_translation = df_states.drop_duplicates().reset_index(drop=True)
    state_space_translation["s_type"] = state_space_translation["s_car_type"]
    state_space_translation["s_age"] = state_space_translation["s_car_age"]
    # convert -1 to 0: s = (0,0) will signify the outside option
    state_space_translation["s_type"] = state_space_translation["s_type"].replace(-1, 0)
    state_space_translation["s_age"] = state_space_translation["s_age"].replace(-1, 0)
    state_space_translation.set_index(cols_s, inplace=True)

    return state_space_translation



def translate_decision_indices(df_decisions):
    """
    df_decisions: a df containing the decision indices present in the data
    """
    # doing some assertion checks
    assert (
        df_decisions.columns == ["d_car_type", "d_car_age"]
    ).all(), "df_decisions must have columns d_car_type and d_car_age"
    assert df_decisions["d_car_type"].min() >= -2, "d_car_type must be >= -2"
    assert df_decisions["d_car_age"].min() >= -2, "d_car_age must be >= -2"
    assert df_decisions["d_car_age"].max() <= 24, "d_car_age must be <= 24"

    ## Translate indices
    d_own_keep = 0
    d_own_purge = 1
    d_own_trade = 2

    decision_space_translation = df_decisions.drop_duplicates().reset_index(drop=True)

    decision_space_translation["d_own"] = -1  # initialize
    decision_space_translation["d_type"] = decision_space_translation["d_car_type"]
    decision_space_translation["d_age"] = decision_space_translation["d_car_age"]

    I = decision_space_translation["d_car_type"] == -1  # keep
    decision_space_translation.loc[I, ["d_own", "d_type", "d_age"]] = [d_own_keep, 0, 0]
    I = decision_space_translation["d_car_type"] == -2  # purge
    decision_space_translation.loc[I, ["d_own", "d_type", "d_age"]] = [
        d_own_purge,
        0,
        0,
    ]
    I = decision_space_translation["d_car_type"] > 0  # trade
    decision_space_translation.loc[I, "d_own"] = d_own_trade

    # assertion that we have all possible values
    abar = (
        decision_space_translation["d_age"].max() + 1
    )  # you cannot buy the clunker, so the oldest d_age is abar-1
    J = decision_space_translation["d_type"].max()
    nD = J * abar + 2
    assert (
        decision_space_translation.shape[0] == nD
    ), f"You have {decision_space_translation.shape[0]} decision space elements, but should have {nD}"
    assert (
        decision_space_translation["d_own"]
        .isin([d_own_keep, d_own_purge, d_own_trade])
        .all()
    )

    decision_space_translation.set_index(["d_car_type", "d_car_age"], inplace=True)

    return decision_space_translation
