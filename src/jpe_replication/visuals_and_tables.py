import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import io

def create_ev_plots(est, folders, options):
    """
    Create plots for the EV terms of the model and the estimation.
    """
    comparison_path = folders["in_comparison_results"]
    n_consumer_types = options['n_consumer_types']
    n_car_types = options['n_car_types']

    ev_dummies_model = pd.read_csv(
        comparison_path + "ev_terms_model.csv", header=None
    )
    ev_dummies_model = ev_dummies_model.values.T

    ev_dummies = est.filter(like="ev_dums_", axis=0)

    ev_dummies_est = ev_dummies.loc[:,'Coefficient'].values.reshape(n_consumer_types, int(ev_dummies.shape[0] / n_consumer_types))
    ev_dummies_se = ev_dummies.loc[:,'se'].values.reshape(n_consumer_types, int(ev_dummies.shape[0] / n_consumer_types))
    
    fig, axs = plt.subplots(2, int(n_consumer_types/2), figsize=(18, 12))
    for i in range(n_consumer_types):
        ax = axs[i // 4, i % 4]
        ax.scatter(np.arange(0,101),ev_dummies_est[i, :], label="DDR-E", marker="+")
        ax.scatter(np.arange(0,101), ev_dummies_model[i, :], label="DNFXP", marker="o", facecolors="none", edgecolors='red')
        ax.set_title("Consumer type: {}".format(i))
        ax.legend()
        ax.set_xlabel("State indices")

        # add variance bands
        # upper 
        # lower 
        #ax.plot(np.arange(0,101),ev_dummies_est[i, :]+1.96*ev_dummies_se[i, :])
        #ax.plot(np.arange(0,101),ev_dummies_est[i, :]-1.96*ev_dummies_se[i, :])


    # plt.plot(ev_dummies_est_full.T)
    plt.savefig(folders['out_results'] + "EV terms consumer types.png", dpi=300, bbox_inches="tight")

def create_params_table(est, folders, model_struct_arrays, options):
    comparison_path = folders["in_comparison_results"]

    est["variablename"] = est.reset_index()["level_0"].values
    est["matches"] = est["variablename"].str.findall("([0-9]+|all)")
    est["nmatches"] = est["matches"].apply(lambda x: len(x))

    est = name_placeholder(est, "matches", "nmatches")

    # next step is to reset index and create a new consistent with the matlab estimates

    # Load MLE estimates
    mle_estimates = io.loadmat(comparison_path + "mp_mle_model.mat")

    # utility car type dummies

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
    est = est.rename(columns={"Coefficient": "DDR-E coefficient"})

    # join
    est = est.join(est_matlab, how="outer")

    est = add_eqb_evs_and_summary_stats(
        est=est,
        comparison_path = comparison_path, 
        model_struct_arrays=model_struct_arrays, 
        options=options,
    )
   
    # Creating a table but dropping all EV terms
    table = est.drop("EV term", level=0).reset_index()
    table.round(4).to_markdown(folders['out_results'] + "results comparison.md")
    
    # rename varnames: 
    varnames = {
        "psych_transcost": "$\\phi_\\{",
        "sigma_s": "$\\sigma_s", 
        "mum": "$\\mu_{",
        "u_0": "$u_{0,",
        "u_a":"$u_{a,",
        "u_a_even": "$u^{even}_{a}",
    }
    table = est.drop("EV term", level=0)
    table=table.rename(index=varnames).reset_index()
    table['varname_latex']=table['variablename'] + table['consumer_type']+ ',' + table['car_type'] + "}$"
    
    cols = list(table.columns)
    table = table[[cols[-1]] + cols[:-1]]
    table = table.drop(columns='variablename')

    with open(folders['out_results'] + 'results_comparision.tex', 'w') as f: 
        f.write(table.style.format(precision=3).to_latex())

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

    df = df[["consumer_type", "car_type", 'car_age', "variablename", "Coefficient", "se"]]
    df = df.set_index(
        [
            "variablename",
            "consumer_type",
            "car_type",
            'car_age',
        ]
    )

    return df


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

def add_eqb_evs_and_summary_stats(est, comparison_path, model_struct_arrays, options):
    ev_dummies_model = pd.read_csv(
        comparison_path + "ev_terms_model.csv", header=None
    ).values
    # create an index
    state_space=np.array(model_struct_arrays['state_space'])
    multi_index = pd.MultiIndex.from_tuples([tuple(row.astype(str)) for row in state_space], names=['car_type', 'car_age'])

    ev_model=pd.concat([
            pd.DataFrame(ev_dummies_model[:,i], index=multi_index, columns=['eqb coefficient']) 
            for i in range(0, options['n_consumer_types'])
        ], 
        keys=np.arange(0, options['n_consumer_types']).astype(str),
        names=['consumer_type']
    ).sort_index()
    ev_model['variablename'] = 'EV term'
    ev_model = ev_model.set_index('variablename', append=True)
    ev_model.index=ev_model.index.reorder_levels(order=['variablename', 'consumer_type', 'car_type', 'car_age'])
    
    # set into est 
    est.loc[ev_model.index, 'eqb coefficient'] = ev_model['eqb coefficient']

    # creating summary statistics: 
    idx=pd.IndexSlice
    mean_abs_rel_err_evs = (
        (
        est.loc[idx['EV term'],'DDR-E coefficient'] - est.loc[idx['EV term'],'eqb coefficient'])
        /
        est.loc[idx['EV term'],'eqb coefficient']
    ).abs().mean()
    select_this=est.index.get_level_values(0).unique().tolist()
    select_this.remove('EV term')
    mean_abs_rel_err_flows = (
        (
        est.loc[idx[select_this],'DDR-E coefficient'] - est.loc[idx[select_this],'eqb coefficient'])
        /
        est.loc[idx[select_this],'eqb coefficient']
    ).abs().mean()

    mean_abs_rel_err = (
        (
        est.loc[:,'DDR-E coefficient'] - est.loc[:,'eqb coefficient'])
        /
        est.loc[:,'eqb coefficient']
    ).abs().mean()


    # Now add this to the table: 
    rel_error_ev=pd.DataFrame({'DDR-E coefficient': [mean_abs_rel_err_evs], 'eqb coefficient': [mean_abs_rel_err_evs]}
    ,index=pd.MultiIndex.from_tuples([('evs_error', '', '', '')],
                                    names=['variablename', 'consumer_type', 'car_type', 'car_age'])
    )
    rel_error_flow=pd.DataFrame({'DDR-E coefficient': [mean_abs_rel_err_flows], 'eqb coefficient': [mean_abs_rel_err_flows]}
    ,index=pd.MultiIndex.from_tuples([('util_error', '', '', '')],
                                    names=['variablename', 'consumer_type', 'car_type', 'car_age'])
    )
    rel_error_total=pd.DataFrame({'DDR-E coefficient': [mean_abs_rel_err], 'eqb coefficient': [mean_abs_rel_err]}
    ,index=pd.MultiIndex.from_tuples([('total_error', '', '', '')],
                                    names=['variablename', 'consumer_type', 'car_type', 'car_age'])
    )

    est=pd.concat([est,rel_error_flow, rel_error_ev, rel_error_total])

    return est