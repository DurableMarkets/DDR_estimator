# Dynamic Programming by Regression
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from IPython.display import Latex
from IPython.display import Markdown
from jax import config
from scipy import io as io

sys.path.insert(0, "../../src/")
sys.path.insert(0, "../logit/")
sys.path.insert(0, "../logit/DDR_estimation/")
sys.path.insert(0, "../logit/monte_carlo/")
sys.path.insert(0, "./data/")

import pdb
from jpe_model.update_specs import set_options
from jpe_model.update_specs import set_params
from jpe_model.state_decision_space import setup_state_space, setup_decision_space

import monte_carlo.mctools as mc
from eqb_model.process_model_struct import create_model_struct_arrays
from jpe_model.update_specs import update_params_and_options
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import utils
import monte_carlo.mctools as mc
import model_interface as mi
import data

from pdr_estimation import estimate_pdr, estimate_pdr_experimental
import pickle

# setting up the output directory for figures and tables
out_dir = "./outputs/wls_on_jpe_with_eqb_moments_simpler_model/"


# enable 64 bit precision
config.update("jax_enable_x64", True)


# Use linear_specification = True for linear specification
linear_specification = True

in_path = "./data/8x4_eqb"
file = in_path + "/arrays_prices_options.pickle"

with open(file, "rb") as handle:
    obj = pickle.load(handle)
(
    state_space,
    map_state_to_index,
    price_index,
    used_car_states,
    prices,
    model_struct_arrays,
    options,
) = obj


# options should match that i'm estimating every consumer type independently.
# params, options=update_params_and_options(dict() , dict())

options = dict()
options["num_consumer_types"] = 8
options["max_age_of_car_types"] = np.array([22] * 4)
options["num_car_types"] = 4  # This specification seem to converge

# options['max_age_of_car_types'] = np.array([23] * 4) # np.array([22] * 4) # this one does not but it does not diverge
# TODO: I think there is something off with the purge decisions in the data.


# set up state and decision space indexing
state_space, map_state_to_index, price_index, used_car_states = setup_state_space(
    options
)

model_struct_arrays = create_model_struct_arrays(
    options=options,
)
decision_space, _ = setup_decision_space(options)


# Load settings
# Read JPE data
infile = in_path + "/ccps_all_years.csv"
dat = pd.read_csv(infile)  # index_col=[0,1,2])
# dat.index.names = ['consumer_type', 'car_type', 'car_age']
dat = dat.set_index(["tau", "s_type", "s_age", "d_own", "d_type", "d_age"])

dat_scrap = pd.read_csv(in_path + "/scrap_all_years.csv")  # index_col=[0,1,2])

# load prices:
years = np.arange(1996, 2009).tolist()
indir_non_data_moments = "./data/model_inputs/small_model_scrap_and_price_from_eqb/"
prices_data = data.read_price_data(
    "./data/8x4/", indir_non_data_moments, years, how="custom", like_jpe=False
)
prices["new_car_prices"] = prices_data["new_car_prices"] / 1000
prices["used_car_prices"] = prices_data["used_car_prices"]
# prices['scrap_car_prices'] = prices_data['scrap_car_prices'] / 1000
prices["scrap_car_prices"] = np.array([6.1989, 5.2565, 9.3461, 8.7610])

state_space = model_struct_arrays["state_space"]
decision_space = model_struct_arrays["decision_space"]

# add the sidx index
dat = dat.reset_index()
state_array = dat[["s_type", "s_age"]].values
dat["sidx"] = map_state_to_index[state_array[:, 0], state_array[:, 1]]
dat = dat.set_index(["tau", "s_type", "s_age", "d_own", "d_type", "d_age"])

dat_scrap = dat_scrap.reset_index()
dat_scrap = dat_scrap[dat_scrap["s_age"] <= 22]
state_array = dat_scrap[["s_type", "s_age"]].values

dat_scrap["sidx"] = map_state_to_index[state_array[:, 0], state_array[:, 1]]
dat_scrap = dat_scrap.set_index(["tau", "s_type", "s_age"])

# add the didx index:
didxs = (
    pd.DataFrame(decision_space, columns=["d_own", "d_type", "d_age"])
    .reset_index()
    .rename(columns={"index": "didx"})
    .set_index(["d_own", "d_type", "d_age"])
)
dat = dat.join(didxs)
# reset index and remove some columns

dat = dat.reset_index()[
    ["tau", "s_type", "s_age", "sidx", "didx", "ccp", "count"]
].set_index(["tau", "s_type", "s_age"])
dat.index.names = ["consumer_type", "car_type", "car_age"]
dat_scrap.index.names = ["consumer_type", "car_type", "car_age"]

scrap_probabilities = data.prepare_scrap_data(dat_scrap, options)

# this step loads (and hence overwrite existing scrap probs.) Max likelihood predictions for scrap probabilities
scrap_probabilities = pd.read_csv(
    indir_non_data_moments + "scrap_probabilities_model.csv", header=None
)
scrap_probabilities = scrap_probabilities.values.T
#breakpoint()
# I want to remove age 23 and 24 from the scrap probabilities
# FIXME: This has bug potential!
clunkers_idx = np.arange(0, 101)[scrap_probabilities[0, :] == 1.0]
indices_to_delete = [idx - (i + 1) for idx in clunkers_idx for i in range(3)]
indices_to_keep = np.delete(np.arange(scrap_probabilities.shape[1]), indices_to_delete)
scrap_probabilities = scrap_probabilities[:, indices_to_keep]

# decisions = model_struct_arrays['decision_space']
# states = model_struct_arrays['state_space']

params = {
    "disc_fac": 0.95,
    "pnew": prices["new_car_prices"],
    "transcost": 0.0,
    "ptranscost": 0.0,
    # "mum2sigma": 0.5,
    "pscrap": np.array([6.1989, 5.2565, 9.3461, 8.7610]),
    "acc_0": np.array([-4.86395656, -4.86395656, -4.86395656, -4.86395656]),
    "acc_a": np.array([0.0, 0.0, 0.0, 0.0]),
    "acc_even": np.array([0.0, 0.0, 0.0, 0.0]),
}

# params, options=update_params_and_options(params, options)

# Trying to replace scrap prices with discounted prices for the clunker
# params['pscrap']=np.array(prices['used_car_prices'][prices['used_prices_indexer'][1:,24]])
# prices['scrap_car_prices']=None#np.array(prices['used_car_prices'][prices['used_prices_indexer'][1:,24]])
# params = namedtuple("model_parameters_to_estimate", params.keys())(**params)
# params
# converts every element in model_struct_arrays


### MODEL SPECIFICATION ###

specification = {
    "mum": (8, 1),
    "buying": (1, 1),
    "scrap_correction": (1, 1),
    "u_0": (1, 4),
    "u_a": (1, 4),
    "u_a_sq": None,
    "u_a_even": None,
}

##### PROCESSING #####

model_struct_arrays = {k: np.array(v) for k, v in model_struct_arrays.items()}
ests = []
nexc = 0
tau_types = dat.index.get_level_values("consumer_type").unique().values

nS = model_struct_arrays["state_space"].shape[0]
nD = model_struct_arrays["decision_space"].shape[0]

# all combinations of decisions and states
combinations = [
    (d, s)
    for d in model_struct_arrays["decision_space"]
    for s in model_struct_arrays["state_space"]
]


# reformatting dta
# TODO: Should be done before receiving the dta
# TODO: Fix  cccp to counts hack
dat.reset_index(inplace=True)

dat = dat.rename(
    columns={
        "consumer_type": "consumer_type",
        "sidx": "state_idx",
        "didx": "decision_idx",
        "count": "counts",  # THIS is a hack for now....
    },
)

sim_df = dat[["consumer_type", "state_idx", "decision_idx", "counts"]].copy()
# debugging options to make it easier to find errors
# sim_df = sim_df.loc[sim_df.consumer_type == 1]

sim_df["consumer_type"] = sim_df["consumer_type"] - 1

# creating variables that are independent of the sample and only depend on model structure:
tab_index = utils.create_tab_index(model_struct_arrays, options)

# creating feasibility index
I_feasible = mi.feasible_choice_all(
    tab_index.get_level_values("state").values,
    tab_index.get_level_values("decision").values,
    state_decision_arrays=model_struct_arrays,
    params=params,
    options=options,
)

# creating data independent regressors
X_indep, model_specification = utils.create_data_independent_regressors(
    tab_index=tab_index,
    prices=prices,
    state_decision_arrays=model_struct_arrays,
    params=params,
    options=options,
    specification=specification,
)

# Construct cfps
cfps = mc.calculate_cfps_from_df(sim_df, calc_counts=False)
counts = mc.calculate_cfps_from_df(sim_df, calc_counts=True)

# create data dependent regressors
X_dep, _ = utils.create_data_dependent_regressors(
    tab_index=tab_index,
    prices=prices,
    scrap_probabilities=scrap_probabilities,
    state_decision_arrays=model_struct_arrays,
    params=params,
    options=options,
    specification=specification,
)

# combine data-independent and data-dependent regressors
X = utils.create_regressors_combine_parts(X_indep, X_dep, model_specification)

# to make it consistent
# FIXME: Explore why cfps end up being of shape smaller than X. (Why do the moments disappear?)
X = X[X.index.isin(cfps.index)]
cfps = cfps[cfps.index.isin(X.index)]

# Estimate pdr regression
est = mc.wls_regression_mc(
    ccps=cfps,
    counts=counts,
    X=X,
    model_specification=model_specification,
)

print(
    est.loc[
        [
            "price_all_0",
            "price_all_1",
            "price_all_2",
            "price_all_3",
            "price_all_4",
            "price_all_5",
            "price_all_6",
            "price_all_7",
        ]
    ]
)

# diagnostic plots
est.filter(like="ev_dums_", axis=0).plot()


# some utilities - Should be put in some auxilliary file at some point


def map_state_from_full_to_reduced(
    sidx_full, state_space_full, map_state_to_index_reduced
):
    try:
        s_idx_reduced = map_state_to_index_reduced[
            state_space_full[sidx_full, 0], state_space_full[sidx_full, 1]
        ]
    except:
        s_idx_reduced = None

    return s_idx_reduced


def from_python_decision_index_to_matlab_decision_index(decision_space_full):
    new_cars = (decision_space_full[..., 0] == 2) & (decision_space_full[..., 2] == 0)
    new_car_idxs = list(np.arange(0, decision_space_full.shape[0])[new_cars])
    oneyear_cars = (decision_space_full[..., 0] == 2) & (
        decision_space_full[..., 2] == 1
    )
    oneyear_car_idxs = list(np.arange(0, decision_space_full.shape[0])[oneyear_cars])

    matlab_index = list(np.arange(0, decision_space_full.shape[0]))

    for new_car_idx, oneyear_car_idx in zip(new_car_idxs, oneyear_car_idxs):
        matlab_index.remove(new_car_idx)
        matlab_index.insert(oneyear_car_idx, new_car_idx)

    return decision_space_full[matlab_index, ...]


def map_decision_from_full_to_reduced(didx_full, matlab_decision_index):
    # new car decisions are located differently in the matlab code vs the python code
    d_type, d_car_type, d_car_age = matlab_decision_index[didx_full]

    # Find the index of the decision in the reduced space

    d_boolean_reduced = (
        (decision_space[:, 0] == d_type)
        & (decision_space[:, 1] == d_car_type)
        & (decision_space[:, 2] == d_car_age)
    )
    decision_idxs_reduced = np.arange(0, decision_space.shape[0])

    decision_idx_reduced = decision_idxs_reduced[d_boolean_reduced]

    if decision_idx_reduced.size == 1:
        decision_idx_reduced = decision_idx_reduced[0]
    else:
        decision_idx_reduced = None

    return decision_idx_reduced


# comparing data ccps

options_full = {}
options_full["num_consumer_types"] = 8
options_full["num_car_types"] = 4
options_full["max_age_of_car_types"] = np.array([25, 25, 25, 25])
state_space_full, _, _, _ = setup_state_space(options_full)


compare_ccps = False
if compare_ccps:
    options_full = {}
    options_full["num_consumer_types"] = 8
    options_full["num_car_types"] = 4
    options_full["max_age_of_car_types"] = np.array([25, 25, 25, 25])
    state_space_full, _, _, _ = setup_state_space(options_full)
    decision_space_full, _ = setup_decision_space(options_full)

    matlab_decision_index_full = from_python_decision_index_to_matlab_decision_index(
        decision_space_full
    )

    matlab_ccps = pd.read_csv(
        indir_non_data_moments + "choice_probabilities_data.csv", header=None
    ).values
    matlab_ccps = matlab_ccps.reshape(8, 101, 102)

    # plotting the difference
    for c_idx in range(8):
        for d in range(102):
            ax = plt.gca()

            values_reduced = np.zeros(101) + np.nan
            values_full = np.zeros(101) + np.nan

            d_reduced = map_decision_from_full_to_reduced(d, matlab_decision_index_full)
            for s in range(101):
                values_full[s] = matlab_ccps[c_idx, s, d]
                s_reduced = map_state_from_full_to_reduced(
                    s, state_space_full, map_state_to_index
                )
                if s_reduced is None or d_reduced is None:
                    continue
                else:
                    try:
                        values_reduced[s] = cfps.loc[c_idx, d_reduced, s_reduced]
                    except:
                        continue

            plt.title(
                "consumer type: {}, decision: {}".format(
                    c_idx, matlab_decision_index_full[d]
                )
            )
            plt.plot(values_reduced, label="ddr ccps")
            plt.xlabel("states")
            plt.plot(values_full, label="matlab ccps")
            plt.legend()
            plt.show()


# Do the same plot with counts
# I'll this for now

# Next compare EV dummies from both models
# Load model EV dummies:
ev_dummies_model = pd.read_csv(
    indir_non_data_moments + "ev_terms_model.csv", header=None
)
ev_dummies_model = ev_dummies_model.values.T

ev_dummies_est = est.filter(like="ev_dums_", axis=0)
ev_dummies_est = ev_dummies_est.values.reshape(8, int(ev_dummies_est.shape[0] / 8))

ev_dummies_est_full = np.zeros((8, 101)) + np.nan

for s in range(101):
    s_reduced = map_state_from_full_to_reduced(s, state_space_full, map_state_to_index)
    if s_reduced is None:
        continue
    else:
        try:
            ev_dummies_est_full[:, s] = ev_dummies_est[:, s_reduced]
        except:
            continue

fig, axs = plt.subplots(2, 4, figsize=(18, 12))
for i in range(8):
    ax = axs[i // 4, i % 4]
    ax.plot(ev_dummies_est_full[i, :], label="wddr", marker="+")
    ax.plot(ev_dummies_model[i, :], label="eqb")
    ax.set_title("Consumer type: {}".format(i))
    ax.legend()
    ax.set_xlabel("State indices")

# plt.plot(ev_dummies_est_full.T)
plt.savefig(out_dir + "EV terms consumer types.png", dpi=300, bbox_inches="tight")
# plt.show()


# plt.plot(ev_dummies_model.T)
# plt.show()

pd.set_option("display.max_rows", None)
# Next compare estimates:


def name_placeholder(df, col_matches, col_nmatches):
    df["consumer_type"] = np.nan
    df["car_type"] = np.nan
    df["car_age"] = np.nan
    for idx in df.index:
        if df.loc[idx, col_nmatches] == 1:
            df.loc[idx, "consumer_type"] = df.loc[idx, col_matches][0]
        elif df.loc[idx, col_nmatches] == 2:
            df.loc[idx, "consumer_type"] = df.loc[idx, col_matches][-1]
            df.loc[idx, "car_type"] = df.loc[idx, col_matches][0]
        elif df.loc[idx, col_nmatches] == 3:
            df.loc[idx, "consumer_type"] = df.loc[idx, col_matches][-1]
            df.loc[idx, "car_type"] = df.loc[idx, col_matches][0]
            df.loc[idx, "car_age"] = df.loc[idx, col_matches][-2]
        else:
            continue

    df["consumer_type"] = df["consumer_type"].fillna("all")
    df["car_type"] = df["car_type"].fillna("all")

    # renaming indices
    df["variablename"] = df["variablename"].replace("price_.*", "mum", regex=True)
    df["variablename"] = df["variablename"].replace(
        "car_type_([0-9]+|all)_([0-9]+|all).*", "u_0", regex=True
    )
    df["variablename"] = df["variablename"].replace(
        "car_type_([0-9]+|all)+_x.*", "u_a", regex=True
    )
    df["variablename"] = df["variablename"].replace("ev_dums_.*", "EV term", regex=True)
    df["variablename"] = df["variablename"].replace(
        "scrap_correction.*", "sigma_s", regex=True
    )
    df["variablename"] = df["variablename"].replace(
        "buying.*", "psych_transcost", regex=True
    )

    df = df[["consumer_type", "car_type", "variablename", "Coefficient"]]
    df = df.set_index(
        [
            "variablename",
            "consumer_type",
            "car_type",
        ]
    )

    return df


# extracting a new index for parameters (FIXME: Some system of keeping track of coeffficients should be implemented and consolidated across the code)
# est['variablename_reversed']=est
est["variablename"] = est.reset_index()["level_0"].values
est["matches"] = est["variablename"].str.findall("([0-9]+|all)")
est["nmatches"] = est["matches"].apply(lambda x: len(x))

est = name_placeholder(est, "matches", "nmatches")

# next step is to reset index and create a new consistent with the matlab estimates

# Load MLE estimates
mle_estimates = io.loadmat(indir_non_data_moments + "mp_mle_model.mat")

# utility car type dummies

def extract_coefficients_from_struct(struct, varname):
    coeffs = struct[varname]
    coeffs_shape = coeffs.shape
    coeffs = coeffs.flatten()
    if (coeffs_shape[0] == 1) & (coeffs_shape[1] == 1):
        coeffs = np.array([elem for elem in coeffs]).reshape(coeffs_shape)
    else:
        coeffs = np.array([elem[0][0] for elem in coeffs]).reshape(coeffs_shape)

    coeffs = pd.DataFrame(coeffs)
    coeffs = coeffs.unstack()
    coeffs.index = coeffs.index.set_names(["car_type", "consumer_type"])
    coeffs = coeffs.reset_index()
    coeffs["variablename"] = varname

    coeffs["car_type"] = coeffs["car_type"] + 1

    if (coeffs_shape[0] == 1) & (coeffs_shape[1] == 1):
        coeffs["car_type"] = "all"
        coeffs["consumer_type"] = "all"
    elif coeffs_shape[0] == 1:
        coeffs["consumer_type"] = "all"
    elif coeffs_shape[1] == 1:
        coeffs["car_type"] = "all"

    else:
        pass

    # convert indices consumer_type and car_type to string
    coeffs["consumer_type"] = coeffs["consumer_type"].astype(str)
    coeffs["car_type"] = coeffs["car_type"].astype(str)

    # Setting new index
    coeffs = coeffs.set_index(
        [
            "variablename",
            "consumer_type",
            "car_type",
        ]
    ).rename(columns={0: "eqb coefficient"})

    return coeffs


def remove_duplicated_values_from_coeffficients(df):
    identical_on_consumer_type = df.groupby(level=1).nunique() == 1
    identical_on_car_type = df.groupby(level=2).nunique() == 1

    if identical_on_consumer_type.all().all():
        df = df.reset_index(level="car_type", drop=True).drop_duplicates()
        df["car_type"] = "all"
        df = df.reset_index().set_index(["variablename", "consumer_type", "car_type"])

    if identical_on_car_type.all().all():
        df = df.reset_index(level="consumer_type", drop=True).drop_duplicates()
        df["consumer_type"] = "all"
        df = df.reset_index().set_index(["variablename", "consumer_type", "car_type"])

    return df


u_0 = extract_coefficients_from_struct(mle_estimates, "u_0")
u_0 = remove_duplicated_values_from_coeffficients(u_0)

# utility car type  age coefficient
u_a = extract_coefficients_from_struct(mle_estimates, "u_a")
u_a = remove_duplicated_values_from_coeffficients(u_a)

# marginal utility of money
mum = extract_coefficients_from_struct(mle_estimates, "mum")

sigma_s = extract_coefficients_from_struct(mle_estimates, "sigma_s")

psych_transcost = extract_coefficients_from_struct(mle_estimates, "psych_transcost")
psych_transcost = remove_duplicated_values_from_coeffficients(psych_transcost)


est_matlab = pd.concat(
    [
        u_0,
        u_a,
        mum,
        sigma_s,
        psych_transcost,
    ],
    axis=0,
)

# rename est
est = est.rename(columns={"Coefficient": "wddr coefficient"})

# join
est = est.join(est_matlab, how="outer")

# Creating a table but dropping all EV terms
table = est.drop("EV term", level=0).reset_index()


table.round(4).to_markdown(out_dir + "results comparison.md")
table.round(4).astype(str).to_latex(out_dir + "results comparison.tex", index=False)



# Next compare ccp predictions from both models:
