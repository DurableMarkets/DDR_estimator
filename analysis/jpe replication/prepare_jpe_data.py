import logit.ddr_tools.main_index as main_index
import jpe_replication.process_data.choice as choice
import jpe_replication.process_data.scrap as scrap
import jpe_replication.process_data.prices as prices
import numpy as np
from data_setups.jpe_options import get_model_specs
import eqb
from eqb.equilibrium import (
    create_model_struct_arrays,
)

# model setup
jpe_model = eqb.load_models("jpe_model")

(params_update, 
 options_update, 
 specification, 
 pricing_options, 
 scrap_options, 
 folders, 
 kwargs) = get_model_specs()

params, options = jpe_model["update_params_and_options"](
    params=params_update, options=options_update
)

# Load settings
model_struct_arrays = create_model_struct_arrays(
    options=options,
    model_funcs=jpe_model,
)

# load main indexer
main_df = main_index.create_main_df(
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
)

# Create choice data 
choice.process_and_reformat_choice_data(
    model_struct_arrays=model_struct_arrays,
    main_df=main_df,
    folders=folders,
    years=kwargs['years'],
    max_age_car=kwargs['max_age_car'],
)

# Create scrap data
scrap.get_scrap_data_from_options(
    scrap_options=scrap_options, 
    model_struct_arrays=model_struct_arrays, 
    main_df=main_df)

# Create price data 
prices.get_price_data_from_options(
    pricing_options=pricing_options, 
    model_struct_arrays=model_struct_arrays, 
    main_df=main_df,
)

