import jax.numpy as jnp
import logit.ddr_tools.utility_helpers as utility_helpers

import numpy as np
import pandas as pd
from example_models.jpe_model.laws_of_motion import calc_accident_probability
import logit.ddr_tools.utility_specs as utility_specs
import logit.ddr_tools.iota_space as iota_space 

def create_data_independent_regressors(
    main_df, prices, model_struct_arrays, model_funcs, params, options, specification
):
    iota = iota_space.create_iota_df(
        feasible_idx=main_df.index, 
        model_struct_arrays=model_struct_arrays, 
        model_funcs=model_funcs, 
        params=params,
        options=options,)

    model_specification = create_model_specification(
        specification, options, model_struct_arrays
    )

    buying, _ = utility_specs.create_buying(
        main_df,
        model_struct_arrays,
        params,
        options,
        specification,
    )

    u_0, _ = utility_specs.create_u_0(
        main_df,
        model_struct_arrays,
        params,
        options,
        specification,
    )

    u_a, _ = utility_specs.create_u_a(
        main_df,
        model_struct_arrays,
        params,
        options,
        specification,
    )

    u_a_sq, _ = utility_specs.create_u_a_sq(
        main_df,
        model_struct_arrays,
        params,
        options,
        specification,
    )

    u_a_even, _ = utility_specs.create_u_a_even(
        main_df,
        model_struct_arrays,
        params,
        options,
        specification,
    )

    # Combine all the flow variables
    X_indep = pd.concat([buying, u_0, u_a, u_a_sq, u_a_even, iota], axis=1)
    breakpoint()
    return X_indep, model_specification


def create_model_specification(specification, options, model_struct_arrays):
    # mum
    _, mum_cols, _ = utility_helpers.construct_utility_colnames(
        "mum", "price_{}_{}", specification, options
    )

    # psych_trans_cost
    _, buying_cols, _ = utility_helpers.construct_utility_colnames(
        "buying", "buying_{}_{}", specification, options
    )

    # scrap_correction
    _, scrap_correction_cols, _ = utility_helpers.construct_utility_colnames(
        "scrap_correction", "scrap_correction_{}_{}", specification, options
    )

    # u_0
    _, u_0_cols, _  = utility_helpers.construct_utility_colnames(
        "u_0", "car_type_{}_{}", specification, options
    )

    # u_a
    _, u_a_cols, _  = utility_helpers.construct_utility_colnames(
        "u_a", "car_type_{}_x_age_{}", specification, options
    )

    # u_a_sq
    _, u_a_sq_cols, _  = utility_helpers.construct_utility_colnames(
        "u_a_sq", "car_type_{}_x_age_sq_{}", specification, options
    )

    # u_a_even
    _, u_a_even_cols, _  = utility_helpers.construct_utility_colnames(
        "u_a_even", "car_type_{}_x_age_even_{}", specification, options
    )

    # Iota
    _, ev_cols = iota_space.create_ev_cols(model_struct_arrays, options)

    model_specification = (
        mum_cols
        + buying_cols
        + scrap_correction_cols
        + u_0_cols
        + u_a_cols
        + u_a_sq_cols
        + u_a_even_cols
        + ev_cols
    )

    return model_specification

