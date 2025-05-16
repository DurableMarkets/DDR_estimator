# Convert counts data for regression 
import numpy as np 
import pandas as pd 
import os 
from jpe_replication.process_data.helpers import read_file
from jpe_replication.process_data.jpe_specific_format_tools.format_tools import (
    translate_state_indices, 
    translate_decision_indices
)
pd.set_option('display.max_rows', 1000)

def get_scrap_data_from_options(scrap_options, model_struct_arrays, main_df):
    scrap_data_source=scrap_options['how']

    if scrap_data_source == 'scrap_data':
        use_scrap_data(
            model_struct_arrays=model_struct_arrays,
            main_df=main_df,
            folders=scrap_options['folders'],
            years=scrap_options['years'],
        )
    elif scrap_data_source == 'model_moments':
        use_moments(scrap_options)
    
    else: 
        raise ValueError(f'{scrap_data_source} is not a valid choice of data source')


def use_scrap_data(
        model_struct_arrays,
        main_df,
        folders: dict,
        years: list = np.arange(1996,2009).astype(str).tolist(),
    ):
    """
    First processes data as it looks like in the JPE data set based on the raw data.
    Then reformats the data such that it fits into the setup in this repo.
    """
    
    dat_scrap=process_scrap_data(
        indir=folders['in_data'], 
        outdir=folders['out_data'], 
        years=years, 
    )
    # storing data
    dat_scrap.to_csv(folders['out_data'] + 'scrap_all_years.csv', index=True)
    print(f'scrap data saved to file scrap_all_years.csv at {folders['out_data']}')

    dat_scrap = reformat_scrap_data(
        in_path=folders['out_data'],
        model_struct_arrays=model_struct_arrays,
    )
    # Merge onto main_df index
    main_df=main_df[[]].reset_index().set_index(['consumer_type', 'state'])
    main_df = main_df.loc[~main_df.index.duplicated(keep='first')][[]]
    dat_scrap = main_df.join(dat_scrap, how='left')
    dat_scrap = dat_scrap[['scrap_counts', 'counts', 'scrap_prob']]
    # set all clunker states to 1.0
    clunker_states = np.unique(model_struct_arrays['clunker_idx'])
    clunker_states =  clunker_states[clunker_states > -1] # removes -9999
    dat_scrap.loc[ pd.IndexSlice[:, clunker_states], :] = 1.0
    # set the no car state to 0.0
    no_car_state = model_struct_arrays['state_index_func'](car_type_state=0,car_age_state=0)
    dat_scrap.loc[ pd.IndexSlice[:, no_car_state], :] = 0.0
    # remaining nans are padded with 0.0
    dat_scrap[dat_scrap.isna()] = 0.0

    # store the data
    dat_scrap.to_pickle(folders['out_data'] + 'scrap_all_years_reformatted.pkl')
    print(f'scrap data saved to file scrap_all_years_reformatted.pkl at {folders['out_data']}')


def process_scrap_data(
        indir, 
        outdir, 
        years: list = np.arange(1996,2009).astype(str).tolist(), 
        #max_age_car: int = 22,
    ):
    # read data
    dat_scrap = read_scrap_data(indir=indir, aggregate=True)

    # construct indices 
    state_space_translation = translate_state_indices(dat_scrap[['s_car_type', 's_car_age']])
   
    # This just seems strangely redundant to me...
    dat_scrap=(dat_scrap.set_index(['tau','s_car_type', 's_car_age']).join(state_space_translation)
    .reset_index('tau') 
    .reset_index(drop=True) 
    .set_index(['tau', 's_type', 's_age'])
    .sort_index()
    )

    # removing decisions and states less than max_age_car (Is this truly used in the JPE paper?)
    max_age_car = 22
    I = (dat_scrap.index.get_level_values('s_age') <= max_age_car+1)
    print(f'Removing {np.sum(~I)} states with age > {max_age_car} in scrap data')
    dat_scrap = dat_scrap.loc[I]

    return dat_scrap


def read_scrap_data(indir, aggregate: bool = False):
    dta_scrap = read_file(indir + f"counts_scrap.xlsx")
    dta_scrap = dta_scrap.rename(
        columns={
            "s_car_type": "s_car_type",
            "s_car_age": "s_car_age",
        }
    )
    dta_scrap = dta_scrap.set_index(["year", "tau", "s_car_type", "s_car_age"])

    # There is these random counts for the no car state. Do people scrap in the no car state or what is this?
    # I just drop them for now
    # It seems like the count is the denominator.
    dta_scrap.drop(
        index=dta_scrap.loc[pd.IndexSlice[:, :, -1, -1], :].index, inplace=True
    )

    dta_scrap["count_scrap"] = dta_scrap["count"] * dta_scrap["pr_scrap"]

    if aggregate:
        dta_scrap = dta_scrap.groupby(level=["tau", "s_car_type", "s_car_age"])[
            ["count_scrap", "count"]
        ].sum()
        dta_scrap["pr_scrap"] = dta_scrap["count_scrap"] / dta_scrap["count"]
    
    dta_scrap.reset_index(inplace=True)
    
    return dta_scrap    

def reformat_scrap_data(in_path, model_struct_arrays):
    dat_scrap = pd.read_csv(in_path + "/scrap_all_years.csv")  # index_col=[0,1,2])
    dat_scrap = dat_scrap[dat_scrap["s_age"] <= 22]
    dat_scrap['state'] = model_struct_arrays['state_index_func'](
        car_type_state=dat_scrap['s_type'].values, 
        car_age_state=dat_scrap['s_age'].values,
    )

    dat_scrap = dat_scrap.rename(columns={"tau": "consumer_type"})
    dat_scrap['consumer_type'] = dat_scrap['consumer_type'] - 1
    dat_scrap = dat_scrap.set_index(['consumer_type','state'])

    dat_scrap.rename(columns={
        "pr_scrap": "scrap_prob",
        "count_scrap": "scrap_counts",
        "count": "counts",
    }, inplace=True)
    
    return dat_scrap

def use_moments(scrap_options):
    file_path = scrap_options['data_source_path']
    
    # load the data 
    dat_scrap = pd.read_csv(file_path, header=None)
    # reformat
    dat_scrap = dat_scrap.unstack()
    dat_scrap.index.names = ['consumer_type', 'state']

    # Conciling formats. 
    dat_scrap=dat_scrap.to_frame(name='scrap_prob')
    dat_scrap['scrap_counts'] = dat_scrap['scrap_prob']
    dat_scrap['counts'] = 1

    # store the data 
    dat_scrap.to_pickle(scrap_options['out_path'] + 'scrap_all_years_reformatted.pkl')
    print(f'scrap from moments saved to file scrap_all_years_reformatted.pkl at {scrap_options['out_path']}')



