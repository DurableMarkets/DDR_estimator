
def calculate_cfps_from_df(df):
    df = df.set_index(["consumer_type", "state", "decision"])
    # denom = df.groupby(["consumer_type", "state_idx"]).counts.sum()
    denom = df.groupby(["consumer_type", "state"]).counts.transform("sum")
    num = df.groupby(["consumer_type", "state", "decision"]).counts.sum()

    df['counts'] = num
    df['ccps'] = num / denom

    df = df.loc[:, ['counts', 'ccps']]
    return df['ccps'], df['counts']


def true_ccps(model_solution):
    assert (
        model_solution["ccps_tau"].shape[0] == 1
    ), "This function only works for one consumer type"
    ccps = model_solution["ccps_tau"][0, :, :]

    return ccps


def calculate_scrap_probabilities(df):
    """
    returns: consumer_type x states Array scrappage probabilities at each state
    estimated from scrap_counts from df.
    """

    df = df.set_index(["consumer_type", "state"])
    df = df.groupby(level=["consumer_type", "state"]).sum()

    df["scrap_prob"] = df["scrap_counts"] / df["counts"]

    scrap_probabilities = df["scrap_prob"].unstack(level=1).values

    return scrap_probabilities