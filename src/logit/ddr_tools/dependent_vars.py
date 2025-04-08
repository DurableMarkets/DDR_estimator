import pandas as pd

def calculate_cfps_from_df(df):
    df = df.set_index(["consumer_type", "state", "decision"])
    # denom = df.groupby(["consumer_type", "state_idx"]).counts.sum()
    denom = df.groupby(["consumer_type", "state"]).counts.transform("sum")
    num = df.groupby(["consumer_type", "state", "decision"]).counts.sum()

    df['counts'] = num
    df['ccps'] = num / denom

    df = df.loc[:, ['counts', 'ccps']]
    return df['ccps'], df['counts']


def true_ccps(main_df, model_solution, options):
    ccps = main_df.copy()

    ccps['ccps'] = model_solution["ccps_tau"][
        ccps.index.get_level_values('consumer_type').values, 
        ccps.index.get_level_values('state').values,
        ccps.index.get_level_values('decision').values,
        ]

    # ccps['ccps_scrap'] = model_solution["ccps_scrap_tau"][
    #     ccps.index.get_level_values('consumer_type').values, 
    #     ccps.index.get_level_values('state').values,
    #     ]


    ccps=ccps.loc[:, "ccps"]

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

def combine_regressors(X_indep, X_dep, model_specification):
    """Combines the data dependent and data independent parts of the X matrix."""
    X = pd.concat([X_indep, X_dep], axis=1)
    X = X.loc[:, model_specification]
    X = X.fillna(0.0)
    return X