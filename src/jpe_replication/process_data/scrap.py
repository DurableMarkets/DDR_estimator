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

def process_scrap_data(
        indir, 
        outdir, 
        years: list = np.arange(1996,2009).astype(str).tolist(), 
        max_age_car: int = 22,
    ):
    # verify that the two dirs exist
    assert os.path.isdir(indir)
    assert os.path.isdir(outdir)

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

    # storing data
    dat_scrap.to_csv(outdir + 'scrap_all_years.csv', index=True)
    print(f'scrap data saved to file scrap_all_years.csv at {outdir}')


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

def reformat_counts_data(indir):
    return None