import numpy as np
import pandas as pd

def create_iota_df(feasible_idx, model_struct_arrays, model_funcs, params, options):
    
    # Create iota matrix
    iota = create_iota_space(feasible_idx, model_struct_arrays, model_funcs, params, options)

    # Create labels
    cols_ev, cols_ev_flat = create_ev_cols(model_struct_arrays, options)

    # Construct a df
    iota_df = pd.DataFrame(
        np.nan,
        index=feasible_idx,
        columns=cols_ev_flat,
    )

    # Set iota onto df
    for tau in range(options["n_consumer_types"]):
        iota_df.loc[pd.IndexSlice[tau, :, :], cols_ev[tau]] = iota
    
    return iota_df


def create_iota_space(feasible_idx, model_struct_arrays, model_funcs, params, options):
    """
    Syntax: create_iota_space(model_struct_arrays, params, options)
    Creates a (ns * nd) x (ns) iota matrix for the model.
    """

    state_transition = create_state_transition_matrix(
        feasible_idx, model_struct_arrays, model_funcs, params, options
    )

    state_dummy_matrix = create_state_dummy_matrix(
        feasible_idx=feasible_idx,
        model_struct_arrays=model_struct_arrays
        )
    
    return params["disc_fac"] * state_transition - state_dummy_matrix


def create_state_transition_matrix(feasible_idx, model_struct_arrays, model_funcs, params, options):
    """
    Syntax: create_state_transition_matrix(model_struct_arrays, params, options)
    Creates a (ns * nd) x (ns) state transition matrix for the model.
    """
    # dimensions
    nS = model_struct_arrays["state_space"].shape[0]  # array of state space
   
    # I need to define the clunker states
    max_age_of_car_types = options["max_age_of_car_types"]
    assert np.all(
        [max_age_of_car_types == max_age_of_car_types[0]]
    ), "Does not work with differing max car ages"
    max_age_of_car_type = max_age_of_car_types[0]
    state_space = model_struct_arrays["state_space"]
    in_car_state = 1
    is_not_clunker = (state_space[:, 1] < max_age_of_car_type) & (
        state_space[:, 0] >= in_car_state
    )
    
    # Consumer types face the same choice so I pick the 
    
    (consumer_types, decisions, states)= (
        feasible_idx.get_level_values(level='tau'), 
        feasible_idx.get_level_values(level='decision'), 
        feasible_idx.get_level_values(level='state'),
    )
    (decisions, states) = (
        decisions[consumer_types == 0], 
        states[consumer_types==0],
    )
    
    post_decision_states = model_struct_arrays["post_decision_state_idxs"][states, decisions]
    next_period_states = model_struct_arrays['next_period_states_idx'][:,1][post_decision_states]

    # construct iota matrix
    fnSD = decisions.shape[0]
    state_transition_matrix = np.zeros((fnSD, nS))

    for i, next_state in enumerate(next_period_states):
        if is_not_clunker[next_state]:
            # Find the clunker state for the specific car.
            car_type = state_space[next_state, 0]
            car_age = state_space[post_decision_states[i], 1] 
            accident_rate = model_funcs['calc_accident_probability'](car_type, car_age, params, options)[
                0
            ]  # returns both acciden rate and 1-accident_rate therefore first element
            clunker_idx = np.arange(nS)[
                (state_space[:, 0] == car_type)
                & (state_space[:, 1] == max_age_of_car_type)
            ]
            state_transition_matrix[i, next_state] = 1 - accident_rate
            state_transition_matrix[i, clunker_idx] = accident_rate
        else:
            state_transition_matrix[i, next_state] = 1
    
    return state_transition_matrix

def create_state_dummy_matrix(feasible_idx, model_struct_arrays):
    """
    Syntax: =create_state_dummy_matrix(model_struct_arrays)
    Creates a (ns * nd) x (ns) dummy matrix that is essentially ns by ns identity matrix repeated nd times
    but insuring that order of states is consistent with the order in the state space object in decision space arrays.
    """

    nS = model_struct_arrays["state_space"].shape[0]  # array of state space
    nD = model_struct_arrays["decision_space"].shape[0]  # array of decision space
    #nSD = nS * nD


    (consumer_types, decisions, states)= (
        feasible_idx.get_level_values(level='tau'), 
        feasible_idx.get_level_values(level='decision'), 
        feasible_idx.get_level_values(level='state'),
    )
    (decisions, states) = (
        decisions[consumer_types == 0], 
        states[consumer_types==0],
    )
    
    fnSD = decisions.shape[0]

    # construct state dummy matrix
    state_dummy_matrix = np.zeros((fnSD, nS))

    for i in range(fnSD):
        state_dummy_matrix[i, states[i]] += 1

    return state_dummy_matrix

def create_ev_cols(state_decision_arrays, options):
    # TODO: I think this is a little dangerous. State_space is of dimension #cartypes*max_car_age + 1 (for no car).
    # So it will match the number of cars since we are removing one car type. But we need a dummy for a new car and not for an old car.
    #
    decisions = state_decision_arrays["decision_space"]
    states = state_decision_arrays["state_space"]
    n_consumer_types = options["n_consumer_types"]

    cols_ev = [
        [
            f"ev_dums_{states[sidx, 0]}_{states[sidx, 1]}_{tau}"
            for sidx in range(states.shape[0])
        ]
        for tau in range(n_consumer_types)
    ]

    cols_ev_flat = [item for sublist in cols_ev for item in sublist]

    return cols_ev, cols_ev_flat
