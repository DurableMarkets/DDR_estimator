# Dynamic Programming by Regression
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import config
from scipy import io as io
import eqb
import jax.numpy as jnp
import jax
import pandas as pd
import matplotlib.pyplot as plt
import logit.ddr_tools.main_index as main_index
import logit.ddr_tools.dependent_vars as dependent_vars
import logit.ddr_tools.regressors as regressors
import logit.prices.prices as dep_prices
import logit.estimators.pwls as pwls
import logit.estimators.npwls as npwls
import logit.estimators.nbinls as nbinls
import logit.estimators.nls as nls

from data_setups.jpe_options import get_model_specs
from set_path import get_paths

from eqb.equilibrium import (
    create_model_struct_arrays,
)
import jpe_replication.process_data.prices as jpe_prices
import jpe_replication.visuals_and_tables as visuals_and_tables
jax.config.update("jax_enable_x64", True)
pd.set_option("display.max_rows", 900)

path_dict = get_paths()

# load model specifications 
params_update, options_update, specification, pricing_options, scrap_options, folders, kwargs = get_model_specs()
jpe_model = eqb.load_models("jpe_model")

### CAN BE REMOVED ###
from scipy import io 
t=io.loadmat("./analysis/data/model_inputs/small_model_scrap_and_price_from_eqb/" + 'mp_mle_model.mat')
#### ##### ##### ####

params, options = jpe_model["update_params_and_options"](
    params=params_update, options=options_update
)

model_struct_arrays=create_model_struct_arrays(
    options=options,
    model_funcs=jpe_model,
)

# load data 
choices=pd.read_pickle(
    './analysis/data/setup_1/processed_data/' + 'ccps_all_years_reformatted.pkl'
)
scraps=pd.read_pickle(
    './analysis/data/setup_1/processed_data/' + 'scrap_all_years_reformatted.pkl'
)
scrap_probabilities = dependent_vars.calculate_scrap_probabilities(scraps.reset_index())

price_dict = pd.read_pickle(
    './analysis/data/setup_1/processed_data/' + 'price_dict.pkl'
)

# create main df
main_df = main_index.create_main_df(
    model_struct_arrays=model_struct_arrays,
    params=params,
    options=options,
)
X_indep, model_specification = regressors.create_data_independent_regressors(
    main_df=main_df,
    prices=price_dict,
    model_struct_arrays=model_struct_arrays,
    model_funcs = jpe_model,
    params=params,
    options=options,
    specification=specification,
)

# create data dependent regressors
X_dep, _ = dep_prices.create_data_dependent_regressors(
    main_df=main_df,
    prices=price_dict,
    scrap_probabilities=scrap_probabilities,
    model_struct_arrays=model_struct_arrays,
    model_funcs=jpe_model,
    params=params,
    options=options,
    specification=specification,
)


# combine independent and dependent regressors
X = dependent_vars.combine_regressors(X_indep, X_dep, model_specification)
cfps, counts = dependent_vars.calculate_cfps_from_df(choices.reset_index())

est, est_post = npwls.owls_regression_mc(
    ccps=cfps, 
    counts=counts, 
    X=X, 
    model_specification=model_specification
)


# Next compare EV dummies from both models
# Load model EV dummies:
visuals_and_tables.create_ev_plots(est, folders)
()
visuals_and_tables.create_params_table(est, folders)
# Compaere estimates



# extracting a new index for parameters (FIXME: Some system of keeping track of coeffficients should be implemented and consolidated across the code)
# est['variablename_reversed']=est
