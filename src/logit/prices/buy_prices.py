import pandas as pd
import numpy as np

def get_price_buy_all(main_df, prices, model_struct_arrays, model_funcs, params):
    decisions = model_struct_arrays["decision_space"]
    nD = decisions.shape[0]

    pbuy_df = main_df.copy()

    p_buy = np.empty(nD)

    for didx in range(nD):
        p_buy[didx] = get_price_buy(didx, prices, model_struct_arrays, model_funcs, params)

    price_buy = pd.DataFrame(
        p_buy, index=pd.Index(range(nD), name="decision", dtype=int),
        columns=["price_buy"]
    )

    pbuy_df['price_buy']=price_buy.loc[pbuy_df.index.get_level_values('decision'),:].values
    pbuy_df = pbuy_df.loc[:, ["price_buy"]]

    return pbuy_df


def get_price_buy(didx, prices, model_struct_arrays, model_funcs, params):
    # assert the prices is a dict
    assert isinstance(prices, dict), f"prices should be a dict but is {type(prices)}"
    assert "used_car_prices" in prices
    assert isinstance(model_struct_arrays, dict)

    decisions = model_struct_arrays["decision_space"]

    d_own, d_type, d_age = decisions[didx, :]
    if (d_own == 0) | (d_own == 1):
        return 0.0
    else:
        if d_age == 0:
            return prices["new_car_prices"][d_type - 1]
        else:
            p_buy = model_funcs['calc_buying_costs'](
                car_type=d_type,
                car_age=d_age,
                params=params,
                used_car_prices=prices["used_car_prices"],
                map_state_to_price_index=prices["used_prices_indexer"],
            )
            # jax -> float
            return float(p_buy)
