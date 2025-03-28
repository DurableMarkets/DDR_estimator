# Convert counts data for regression 
import numpy as np 
import pandas as pd 
import os 
import data 
pd.set_option('display.max_rows', 1000)

indir  = './data/8x4/'
outdir = './data/8x4_eqb/'

# verify that the two dirs exist
assert os.path.isdir(indir)
assert os.path.isdir(outdir)

# read data
years = np.arange(1996,2009).tolist()
#years = np.arange(2001,2002).tolist()
years = [str(y) for y in years]
prices = data.read_price_data(indir=indir, years=years, how='unweighted')

dat = data.read_data(years=years, indir=indir, read_scrap=False)
dat_scrap = data.read_scrap_data(indir=indir, aggregate=True)

# padding data where nans exist with 2 since they indicate a value of either [1,2,3,4,5]
dat.loc[dat['count'].isna(), 'count'] = 2

# construct indices 
breakpoint()
state_space_translation = data.translate_state_indices(dat[['s_car_type', 's_car_age']])
decision_space_translation = data.translate_decision_indices(dat[['d_car_type', 'd_car_age']])

# setting the index and joining translations on these
dat = dat.set_index(['tau', 's_car_type', 's_car_age', 'd_car_type', 'd_car_age'])

dat=(dat
 .join(decision_space_translation, on=['d_car_type', 'd_car_age'])
 .join(state_space_translation, on=['s_car_type', 's_car_age'])
 .reset_index('tau') # keep this 
 .reset_index(drop=True) # drop old indices 
 .set_index(['tau', 's_type', 's_age'])#.set_index(['tau', 's_type', 's_age', 'd_own', 'd_type', 'd_age'])
)
dat_scrap=(dat_scrap.join(state_space_translation)
 .reset_index('tau') # keep this 
 .reset_index(drop=True) # drop old indices 
 .set_index(['tau', 's_type', 's_age'])#.set_index(['tau', 's_type', 's_age', 'd_own', 'd_type', 'd_age'])
 .sort_index()
 )
# keep and no car is illegal. Recoding to purge and no_car 
d_own=dat.loc[pd.IndexSlice[:,0,0],'d_own'] # extracts d_own for all no car states across consumer types
d_own= np.where(d_own == 0, 1, d_own)       # replaces 0 with 1 ie. keep with purge
dat.loc[pd.IndexSlice[:,0,0],'d_own']=d_own # sets corrected values of d_own into the no car states

# set index for data 
cols = ['tau','s_type','s_age','d_own','d_type','d_age']
dat = dat.reset_index().set_index(cols)[['count']]

# removing decisions and states less than max_age_car (Is this truly used in the JPE paper?)
max_age_car = 22
I = (dat.index.get_level_values('s_age') < max_age_car) & (dat.index.get_level_values('d_age') < max_age_car)
print(f'Removing {np.sum(~I)} state,decision pairs with age > {max_age_car} in transition data')
dat = dat.loc[I]

I = (dat_scrap.index.get_level_values('s_age') <= max_age_car+1)
print(f'Removing {np.sum(~I)} states with age > {max_age_car} in scrap data')
dat_scrap = dat_scrap.loc[I]

# I'm unsure whether it is in fact post decision states that go into the estimator?

#TODO: make a function that aggregates instead
dat = dat.groupby(cols)['count'].sum().to_frame('count')

# Construct cfps
shares = dat # pointer 
# Nans are 1,2,3,4,5 so we set them 2

shares['count_state'] = shares.groupby(level=['tau', 's_type', 's_age'])['count'].transform('sum')

#shares=shares.join(share_denom, rsuffix='_denom')
shares['ccp'] = shares['count']/shares['count_state']


assert np.isclose(shares['ccp'].groupby(level=[0,1,2]).sum() , 1.0).all()
breakpoint()
# setting the index
shares.to_csv(outdir + 'ccps_all_years.csv', index=True)
dat_scrap.to_csv(outdir + 'scrap_all_years.csv', index=True)

print('Average observations pr. state:')
print(dat['count'].sum()/(8*101)) 
print('Average observations pr. consumer and state:')
print(dat.groupby(['tau'])['count'].sum()/101)
np.round(44447*51)
print(dat['count'].sum())