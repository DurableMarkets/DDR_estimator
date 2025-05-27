import pandas as pd
import numpy as np
import logit.ddr_tools.utility_helpers as utility_helpers
import logit.prices.buy_prices as buy_prices
import logit.prices.sell_prices as sell_prices
import logit.prices.scrap_correction as scrap_correction


def create_data_dependent_regressors(
    main_df,
    prices,
    scrap_probabilities,
    model_struct_arrays,
    model_funcs,
    params,
    options,
    specification,
):
    # Create pricing
    pricing, pricing_cols = create_pricing(
        main_df,
        prices,
        scrap_probabilities,
        model_struct_arrays,
        model_funcs,
        params,
        options,
        specification,
    )

    # Create scrap_correction
    scrap_correct, scrap_correct_cols = create_scrap_correction(
        main_df,
        prices,
        scrap_probabilities,
        model_struct_arrays,
        model_funcs,
        params,
        options,
        specification,
    )

    # Create data dependent regressors
    X_dep = pd.concat([pricing, scrap_correct], axis=1)

    # combine columns
    X_dep_cols = pricing_cols + scrap_correct_cols

    return X_dep, X_dep_cols

def create_pricing(
    main_df,
    prices,
    scrap_probabilities,
    model_struct_arrays,
    model_funcs,
    params,
    options,
    specification,
):
    # create price cols
    price_cols, price_cols_flat, price_cols_looper = utility_helpers.construct_utility_colnames(
        "mum", "price_{}_{}", specification, options
    )

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=price_cols_flat,
    )

    # Buy prices
    pbuy_df = buy_prices.get_price_buy_all(main_df, prices, model_struct_arrays, model_funcs, params)

    # Sell prices
    psell_df = sell_prices.get_price_sell_all(
        main_df, prices, scrap_probabilities, model_struct_arrays, params, options
    )
    
    # creating price differences
    for ntype in range(0, options["n_consumer_types"]):
            X.loc[
                pd.IndexSlice[ntype, :, :, :, :], price_cols_flat[ntype]
            ] = (
                psell_df.loc[pd.IndexSlice[ntype, :, :, :, :], 'price_sell'] 
                - pbuy_df.loc[pd.IndexSlice[ntype, :, :, :, :], 'price_buy']
            )
    X = X.loc[:, price_cols_flat]

    return X, price_cols_flat


def create_scrap_correction(
    main_df,
    prices,
    scrap_probabilities,
    model_struct_arrays,
    model_funcs,
    params,
    options,
    specification,
):


    # create price cols
    scrap_cols, scrap_cols_flat, scrap_cols_looper = utility_helpers.construct_utility_colnames(
        "scrap_correction", "scrap_correction_{}_{}", specification, options
    )

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=scrap_cols_flat,
    )


    scrap_correct = scrap_correction.scrap_correction_all(
        main_df, scrap_probabilities, model_struct_arrays, options
    )

    # if nconsumers == 1:
    #     X.loc[:, scrap_cols_flat[0]] = Z["scrap_correction"]
    # elif nconsumers > 1:
    #     for tau, scrap_col in enumerate(scrap_cols_flat):
    #         idx_tau = pd.IndexSlice[tau, :, :]
    #         X.loc[idx_tau, scrap_col] = Z.loc[idx_tau, "scrap_correction"]

    # scrap correction terms 
    if specification['scrap_correction'][1] > 1:
         raise ValueError('scrap_correction is not allowed to vary with car_type')
    for ntype in range(0, options["n_consumer_types"]):
        for ncartype in range(0, options['n_car_types']+1): # IMPORTANT ZERO IS INCLUDED
            X.loc[
                pd.IndexSlice[ntype, :, :, :, :], scrap_cols_looper[ntype][0]
            ] = (
                scrap_correct.loc[pd.IndexSlice[ntype, :, :, :, :], 'scrap_correction'].values                 
            )
    
    X = X.loc[:, scrap_cols_flat]
       
    return X, scrap_cols_flat
