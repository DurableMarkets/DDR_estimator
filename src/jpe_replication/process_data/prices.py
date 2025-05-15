# Convert counts data for regression 
import numpy as np 
import pandas as pd 
import os 
from jpe_replication.process_data.helpers import read_file
from jpe_replication.process_data.jpe_specific_format_tools.format_tools import (
    translate_state_indices, 
    translate_decision_indices
)
import jax



def read_price_data(indir_jpe_data, indir_moments, years, how, model_struct_arrays, like_jpe=False):
    dta = read_file(indir_jpe_data + f"car_attributes.xlsx")
    years = [int(year) for year in years]
    np_state_decision_arrays = jax.tree_util.tree_map(
        lambda x: np.array(x), model_struct_arrays
    )

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
        #scrap_prices = np.array([0.0, 0.0, 0.0, 0.0])
        prices = {
            "new_car_prices": new_prices,
            "used_car_prices": used_prices,
            "scrap_car_prices": scrap_prices,
            'used_prices_indexer': np_state_decision_arrays["map_state_to_price_index"],
        }

    else:
        raise ValueError(
            f"how must be either weighted, unweighted or custom, but is {how}"
        )

    return prices

