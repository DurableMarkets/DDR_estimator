import os
import numpy as np
import pandas as pd


def read_file(fpath: str):
    breakpoint()
    assert os.path.isfile(fpath), f"File {os.path.basename(fpath)} does not exist"
    counts = pd.read_excel(fpath)
    return counts


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


def read_data(years: list, indir, read_scrap: bool = False):
    """Reads data from years in list years.

    If read_scrap is True, reads scrap data as well.

    """
    dta = read_file(indir + f"counts_{years[0]}.xlsx")
    for y in years[1:]:
        dta = pd.concat([dta, read_file(indir + f"counts_{y}.xlsx")], axis=0)

    # dta = dta.set_index(['year', 'tau', 's_car_type', 's_car_age', 'd_car_type', 'd_car_age'])

    if read_scrap:
        raise NotImplementedError("I have not implemented the scrap data yet")

    return dta


def read_scrap_data(indir, aggregate: bool = False):
    dta_scrap = read_file(indir + f"counts_scrap.xlsx")
    dta_scrap = dta_scrap.rename(
        columns={
            "s_car_type": "s_car_type",
            "s_car_age": "s_car_age",
        }
    )
    dta_scrap = dta_scrap.set_index(["year", "tau", "s_car_type", "s_car_age"])

    # There is these random counts for the no car state. Do people scrap in the no car state or what is this?
    # I just drop them for now
    # It seems like the count is the denominator.
    dta_scrap.drop(
        index=dta_scrap.loc[pd.IndexSlice[:, :, -1, -1], :].index, inplace=True
    )

    dta_scrap["count_scrap"] = dta_scrap["count"] * dta_scrap["pr_scrap"]
    # dta_scrap.drop(columns=['count'], inplace=True)

    if aggregate:
        dta_scrap = dta_scrap.groupby(level=["tau", "s_car_type", "s_car_age"])[
            ["count_scrap", "count"]
        ].sum()
        dta_scrap["pr_scrap"] = dta_scrap["count_scrap"] / dta_scrap["count"]

    return dta_scrap


def read_price_data(indir_jpe_data, indir_moments, years, how, like_jpe=False):
    dta = read_file(indir_jpe_data + f"car_attributes.xlsx")
    years = [int(year) for year in years]

    if how == "weighted":
        raise NotImplementedError("I have not implemented the price data yet")

    elif how == "unweighted":
        # raise NotImplementedError('I have not implemented the price data yet')
        dta = dta.set_index(["car_type", "car_age", "year"])
        dta = dta.loc[pd.IndexSlice[:, :, years], :]

        prices = dta.groupby(level=["car_type", "car_age"]).mean()
        if like_jpe:
            # For some reason the JPE paper uses one year old cars as the new car price instead of 0 year old cars.
            new_car_age = 1
            new_prices = prices.loc[pd.IndexSlice[:, new_car_age], "price_new"].values
        elif not like_jpe:
            new_car_age = 0
            new_prices = prices.loc[pd.IndexSlice[:, new_car_age], "price_new"].values
        
        used_prices = prices.loc[pd.IndexSlice[:, 1:], "price_new"].values
        scrap_prices = np.array([0.0] * 4)
        prices = {
            "new_car_prices": new_prices,
            "used_car_prices": used_prices,
            "scrap_car_prices": scrap_prices,
        }

    elif how == "custom":
        dta = dta.set_index(["car_type", "car_age", "year"])
        dta = dta.loc[pd.IndexSlice[:, :, years], :]
        prices = dta.groupby(level=["car_type", "car_age"]).mean()

        # new prices
        new_car_age = 1
        new_prices = prices.loc[pd.IndexSlice[:, new_car_age], "price_new"].values

        # used prices:
        # used_prices = pd.read_excel(indir + f'ligevaegtspriser.xlsx').values.flatten() # Insert data here
        used_prices = pd.read_csv(
            indir_moments + f"used_car_prices_model.csv", header=None
        ).values.flatten()

        # scrap prices

        scrap_prices = np.array([6.1989, 5.2565, 9.3461, 8.7610])

        prices = {
            "new_car_prices": new_prices,
            "used_car_prices": used_prices,
            "scrap_car_prices": scrap_prices,
        }

    else:
        raise ValueError(
            f"how must be either weighted, unweighted or custom, but is {how}"
        )

    return prices


def prepare_scrap_data(dat_scrap, options):
    """Constructs a matrix of size (Ntypes x(Ncartypes*Ncarages + 1), that mimics the
    format used in simulations."""

    num_consumer_types = options["num_consumer_types"]
    num_car_types = options["num_car_types"]
    max_age_of_car_types = options["max_age_of_car_types"][0]

    prob_scrap = (
        dat_scrap.reset_index()
        .set_index(["consumer_type", "sidx"])["pr_scrap"]
        .unstack(level="sidx")
        .values
    )
    scrap_probabilities = np.ones(
        (num_consumer_types, num_car_types * max_age_of_car_types + 1)
    )
    scrap_probabilities[:, 0 : (num_car_types * max_age_of_car_types)] = prob_scrap

    return scrap_probabilities
