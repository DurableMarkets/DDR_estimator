import numpy as np


# Compute polynomials
def pol(x, degree=2):
    n = x.shape[0]
    xmat = np.zeros((n, degree))
    for i in range(degree):
        xmat[:, i] = x ** (i + 1)

    return xmat


# Create x_ij matrix
def create_x_matrix(
    state_space,
    decision_space,
    state_data,
    w_j_vars=["age_pol", "new", "d1", "keep", "purge"],
    w_j_deg=2,
    s_i_vars=["age_pol", "nocar"],
    s_i_deg=2,
    infeasible_choices=["w_keep*s_nocar"],
    unwanted_choices=None
):
    """Create x_ij matrix for conditional logit model.
    Args:
        state_space: n_states x n_state_variables matrix of state variables
        decision_space: n_decisions x n_decision_variables matrix of decision variables
        state_data: n_obs x 1 vector of state indexes
        w_j_vars: list of variables to include in w_j vars can include
            - 'age_pol': polynomial age of chosen car
            - 'new': dummies for new car
            - 'keep': dummies for keeping
            - 'purge': dummies for purging

        w_j_deg: degree of polynomial age of chosen car (default=2)
                (only relevant if 'age_pol' is in vars)

        s_i_vars: list of variables to include in s_i vars can include
            - 'age_pol': polynomial age of existing car
            - 'nocar': dummy for no car
            - 'state_dummy': dummies for all states

        s_i_deg: degree of polynomial age of existing car (default=2)
                (only relevant if 'age_pol' is in vars)

        infeasible_choices: list of infeasible choices to remove from x_ij, can include
            - 'w_keep*s_nocar': keep and no car is infeasible
            - 'w_trade*s_nocar': trade perfectly predict no car, so trade and no car should not be interacted

    Returns:
        x: n_obs x J x K matrix of explanatory variables
        name: list of names of explanatory variables

    Example:
        x, name = create_x_matrix(state_space, decision_space, state_data,
                w_j_vars=['age_pol','new','keep','purge'],
                w_j_deg=2,
                s_i_vars=['age_pol', 'nocar'],
                s_i_deg=2,
                infeasible_choices= ['w_keep*s_nocar'])
        creates a matrix x with 19 columns with corresponding names given in the list name

        name=[  'w_age_pol(0)*s_age_pol(1)',
                    'w_age_pol(0)*s_age_pol(2)',
                    'w_age_pol(0)*s_nocar',
                    'w_age_pol(1)*s_age_pol(1)',
                    'w_age_pol(1)*s_age_pol(2)',
                    'w_age_pol(1)*s_nocar', 
                    'w_new*s_age_pol(1)',
                    'w_new*s_age_pol(2)',
                    'w_new*s_nocar',
                    'w_trade*s_age_pol(1)',
                    'w_trade*s_age_pol(2)',
                    'w_trade*s_nocar',
                    'w_keep*s_age_pol(1)',
                    'w_keep*s_age_pol(2)',
                    'w_age_pol(0)',
                    'w_age_pol(1)',
                    'w_new',
                    'w_trade',
                    'w_keep']

        Here the first 14 columns are z_ij variables (ie interactions between w_j and s_i),
        the last 5 columns are w_j variables (ie decision specific variables)

        Note that the variable 'w_keep*s_nocar' is an infeasible choice and is therefore removed from x_ij

    """
    # TODO: add dummies for each car type + interaction with age of chosen car. I guess w_age_pol(0)*car_type...
    n_obs = state_data.shape[0]
    s_i, s_name = state_specific_variables(state_space, state_data, s_i_vars, s_i_deg)
    w_j, w_name = decision_specific_variables(decision_space, w_j_vars, w_j_deg)
    w_j = w_j[np.newaxis, :, :].repeat(n_obs, axis=0)

    # Compute z_ij (i.e. w_j \otimes s_i):
    z_ij = np.einsum("ijk,il->ijkl", w_j, s_i).reshape(
        w_j.shape[0], w_j.shape[1], w_j.shape[-1] * s_i.shape[-1]
    )

    # Combine labels for z_ij
    name_zij = []
    for i in range(len(w_name)):
        for j in range(len(s_name)):
            name_zij = name_zij + [w_name[i] + "*" + s_name[j]]

    # create x_ij matrix
    x = np.concatenate((z_ij, w_j), axis=2)

    name = name_zij + w_name

    for infeasible_choice in infeasible_choices:
        try:
            index = name.index(infeasible_choice)
            print(
                f"The variable '{infeasible_choice}' is an infeasible choice - removed from x and name"
            )
            x = np.delete(x, index, axis=2)
            name.remove(infeasible_choice)
        except ValueError:
            print(f"'{infeasible_choice}' is not included - continue.")

    if unwanted_choices is not None:
        for unwanted_choice in unwanted_choices:
            try:
                index = name.index(unwanted_choice)
                print(
                    f"The variable '{unwanted_choice}' unwanted in regression - removed from x and name"
                )
                x = np.delete(x, index, axis=2)
                name.remove(unwanted_choice)
            except ValueError:
                print(f"'{unwanted_choice}' is not included - continue.")


    return x, name


# Function to create state specific variables, s_i
def state_specific_variables(
    state_space, state_data, vars=["age_pol", "nocar"], degree=2
):
    """Construct state specific variables that only vary by state.
    Args:
        state_space: n_states x n_state_variables matrix of state variables
        state_data: n_obs x 1 vector of state indexes
        vars: list of variables to include in s_i vars can include
            - 'age_pol': polynomial age of existing car
            - 'nocar': dummy for no car
            - 'state_dummy': dummies for all states

        degree: degree of polynomial age of existing car (default=2)
                (only relevant if 'age_pol' is in vars)

        Returns:
            s: n_obs x n_s matrix of state specific variables

    Examples:
        s = state_specific_variables(state_space, state_data, vars=['age_pol', 'nocar'], degree=2)
        creates a matrix s with 4 columns, the first three columns contains polynomial age of existing car
            [age/25, (age/25)^2, (age/25)^3], where age is the age of the existing car
            the last column is a dummy for no car

        s = state_specific_variables(state_space, state_data, vars=['state_dummy'])
        creates a matrix s with columns of dummies for all states

    """
    no_car = 0
    n_obs = len(state_data)
    n_states = state_space.shape[0]
    s = None
    name = []
    for var in vars:
        if var == "age_pol":
            # polynomial age of existing car
            s_k = pol(state_space[state_data, 1] / 25, degree)
            name = name + ["s_" + var + "(" + str(i + 1) + ")" for i in range(degree)]
        elif var == "nocar":
            # dummy for no car
            s_k = np.zeros((n_obs, 1))
            s_k[:, 0] = state_space[state_data, 0] == no_car
            name = name + ["s_" + var]
        elif var == "state_dummy":
            # dummies for all states
            s_k = np.zeros((n_obs, n_states))
            s_k[np.arange(n_obs), state_data] = 1
            name = name + ["s_" + var + "(" + str(i) + ")" for i in range(n_states)]
        else:
            raise ValueError(f"Variable {var} not defined")
        if s is None:
            s = np.array(s_k)
        else:
            s = np.concatenate((s, s_k), axis=1)
    return s, name


# Function to create decision specific variables, w_j
def decision_specific_variables(
    decision_space, vars=["age_pol", "new", "trade", "purge"], degree=2
):
    """Construct decision specific variables that only vary by decision.
    Args:
        decision_space: n_decisions x n_decision_variables matrix of decision variables
        vars: list of variables to include in w_j vars can include
            - 'age_pol': polynomial age of chosen car
            - 'new': dummies for new car
            - 'trade': dummies for trading
            - 'purge': dummies for purging

        degree: degree of polynomial age of existing car (default=2)
                (only relevant if 'age_pol' is in vars)

        Returns:
            w: n_obs x n_w matrix of decision specific variables

    Examples:
        w = decision_specific_variables(decision_space, vars=['age_pol', 'new', 'trade', 'purge'], degree=2)
        creates a matrix w with 6 columns, the first three columns contains polynomial age of chosen car
            [age/25, (age/25)^2, (age/25)^3], where age is the age of the chosen car
            the next two columns are dummies for new car and dummies for trading
            the last column is a dummy for purging

        w = decision_specific_variables(decision_space, vars=['new', 'trade', 'purge'])
        creates a matrix w with columns of dummies for new car, trading and purging

    """
    J = decision_space.shape[0]
    n_decisions = decision_space.shape[1]
    keep = 0
    purge = 1
    trade = 2

    w = None
    name = []
    for var in vars:
        if var == "age_pol":
            # polynomial age of chosen car
            w_j = pol(decision_space[:, 2] / 25, degree)
        elif var == "new":
            # dummies for new car
            w_j = np.zeros((J, 1))
            w_j[:, 0] = (decision_space[:, 0] == trade) & (decision_space[:, 2] == 0)
        elif var == "d1":
            # dummies for bying 1 year old car
            w_j = np.zeros((J, 1))
            w_j[:, 0] = (decision_space[:, 0] == trade) & (decision_space[:, 2] == 1)
        elif var == "trade":  # dummies for trading
            w_j = np.zeros((J, 1))
            w_j[:, 0] = decision_space[:, 0] == trade
        elif var == "purge":  # dummy for purging
            w_j = np.zeros((J, 1))
            w_j[:, 0] = decision_space[:, 0] == purge
        elif var == "keep":  # dummy for keeping
            w_j = np.zeros((J, 1))
            w_j[:, 0] = decision_space[:, 0] == keep
        elif var == "all":  # dummies for all decisions
            w_j = np.zeros((J, n_decisions - 1))
            w_j[np.arange(J), decision_space[:, 0]] = 1
            w_j = np.delete(w_j, 0, axis=1)
        else:
            raise ValueError(f"Variable {var} not defined")
        if w is None:
            w = np.array(w_j)
        else:
            w = np.concatenate((w, w_j), axis=1)

        if w_j.shape[1] == 1:
            name = name + ["w_" + var]
        else:
            name = name + ["w_" + var + "(" + str(i) + ")" for i in range(w.shape[1])]

    return w, name
