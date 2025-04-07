import numpy as np

def scrap_correction_all(
    main_df, scrap_probabilities, model_struct_arrays, options
):  # model_struct_arrays, params, prices, options):
    state_space = model_struct_arrays["state_space"]
    nS = state_space.shape[0]
    num_consumer_types = options["n_consumer_types"]

    sc_df = main_df.copy()
    sc_df["dum_hascar"] = sc_df["car_type_state"] != 0
    sc_df["dum_keep"] = sc_df["own_decision"] == 0
    sc_df["dum_getting_rid_of_car"] = sc_df["dum_hascar"] & ~sc_df["dum_keep"]


    assert (
        num_consumer_types == scrap_probabilities.shape[0]
    ), f"num_consumer_types={num_consumer_types} but scrap_probabilities.shape[0]={scrap_probabilities.shape[0]}"

    scrap_correction = np.empty((num_consumer_types, nS))
    for tau in range(num_consumer_types):
        scrap_prob_tau = scrap_probabilities[tau, :]
        scrap_correction[tau, :] = binary_entropy_vec(scrap_prob_tau)

    sc_df['scrap_correction'] = 0.0 # initialize at zero
    I = sc_df["dum_getting_rid_of_car"].values
    
    sc_df.loc[I, "scrap_correction"] = scrap_correction[
        sc_df.loc[I,:].index.get_level_values('consumer_type').values, 
        sc_df.loc[I,:].index.get_level_values('state').values]

    sc_df = sc_df.loc[:, ["scrap_correction"]]

    

    return sc_df


def binary_entropy_vec(p):
    E = np.zeros_like(p)
    I = (p > 0.0) & (p < 1.0)
    E[I] = -p[I] * np.log(p[I]) - (1.0 - p[I]) * np.log(1.0 - p[I])
    return E
