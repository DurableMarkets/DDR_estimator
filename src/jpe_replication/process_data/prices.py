# Convert counts data for regression 
import numpy as np 
import pandas as pd 
import os 
from jpe_replication.process_data.helpers import read_file
from jpe_replication.process_data.jpe_specific_format_tools.format_tools import (
    translate_state_indices, 
    translate_decision_indices
)
import jpe_replication.visuals_and_tables as visuals_and_tables
from jax.tree_util import tree_map
import pickle

def get_price_data_from_options(
        pricing_options, 
        model_struct_arrays, 
        main_df,
    ):
    data_source =pricing_options['how']
    if data_source == 'model_moments':
        new_prices, used_prices, scrap_prices=use_moments(pricing_options)
    else:
        raise ValueError(f'{data_source} is not a valid choice of data source')


    # Consturct lookup tool 
    np_state_decision_arrays = tree_map(
        lambda x: np.array(x), model_struct_arrays
    )

    price_dict = {
        'new_car_prices': new_prices,
        'used_car_prices': used_prices,
        'scrap_car_prices': scrap_prices,
        'used_prices_indexer': np_state_decision_arrays['map_state_to_price_index'],
    }

    # store data - pickle it
    out_file = pricing_options['out_path'] + 'price_dict.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(price_dict, f)
    
def use_moments(pricing_options):
    # new prices 
    new_prices = pricing_options['new_prices']
    
    # used prices:
    used_prices = pd.read_csv(
       pricing_options['data_source_path'], header=None
    ).values.flatten()
    
    # scrap prices
    scrap_prices = pricing_options['scrap_prices']

    return new_prices, used_prices, scrap_prices
