import numpy as np
import jax
def construct_price_dict(equ_price, state_space_arrays, params, options):
    np_state_decision_arrays = jax.tree_util.tree_map(
        lambda x: np.array(x), state_space_arrays
    )

    prices = {
        "used_car_prices": np.array(equ_price),
        "new_car_prices": np.array(params["pnew"]),
        "scrap_car_prices": np.array(params["pscrap"]),
        "used_prices_indexer": np_state_decision_arrays["map_state_to_price_index"],
    }

    return prices