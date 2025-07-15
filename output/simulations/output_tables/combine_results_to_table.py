import pandas as pd

# We want 
load_from = [
    'setup_paper_25/npwls',
    'setup_paper_250/npwls'
]

dfs = []
for load in load_from: 
    #
    df=pd.read_pickle(f'./output/simulations/{load}/mc.pkl')
    N = int(df.loc[pd.IndexSlice['Sample size', :, :, :, : ],'mean'].values[0])
    df['N'] = N
    df = df.set_index('N', append=True)

    dfs.append(df)

df_full = pd.concat(dfs,axis=0)

# Reset index
# age vars are not unique across the 
# car_type
idx=pd.IndexSlice 
df_full.loc[idx['flow','car_type','all','1',:,:],'true value']=df_full.loc[idx['flow','car_type','0','1',:,:],'true value'][0]
df_full.loc[idx['flow','car_type','all','2',:,:],'true value'] = df_full.loc[idx['flow','car_type','0','2',:,:],'true value'][0]
df_full.loc[idx['flow','car_type','all','3',:,:],'true value'] = df_full.loc[idx['flow','car_type','0','3',:,:],'true value'][0]
df_full.loc[idx['flow','car_type','all','4',:,:],'true value'] = df_full.loc[idx['flow','car_type','0','4',:,:],'true value'][0]

# car_type_age
df_full.loc[idx['flow','car_type_age','all','1',:,:],'true value']=df_full.loc[idx['flow','car_type_age','0','1',:,:],'true value'][0]
df_full.loc[idx['flow','car_type_age','all','2',:,:],'true value'] = df_full.loc[idx['flow','car_type_age','0','2',:,:],'true value'][0]
df_full.loc[idx['flow','car_type_age','all','3',:,:],'true value'] = df_full.loc[idx['flow','car_type_age','0','3',:,:],'true value'][0]
df_full.loc[idx['flow','car_type_age','all','4',:,:],'true value'] = df_full.loc[idx['flow','car_type_age','0','4',:,:],'true value'][0]

# buying
df_full.loc[idx['flow','buying',:,:]]

# Building a table:
varnames = ['price','buying', 'car_type', 'car_type_age', 'price', 'scrap_correction']
colnames = ['true value', 'mean', 'MCSE', 'MASE']
tab = df_full.loc[idx['flow', varnames], colnames]
tab= tab.reorder_levels(['N', 'vartype', 'varname', 'consumer_type', 'car_type', 'car_age'])
tab = tab.sort_index()

# Mapping 

latex_notation=pd.DataFrame(index=tab.index, columns=['greek'])

# price
price_greek = lambda c: f'$\\alpha_{c}$'
latex_notation.loc[idx[:,:,'price', :], 'greek'] = [
    price_greek(c) for c in latex_notation.loc[idx[:,:,'price']].index.get_level_values('consumer_type').to_list()
    ]

# scrap_correction
price_greek = lambda c: f'$\\sigma$'
latex_notation.loc[idx[:,:,'scrap_correction', :], 'greek'] = [
    price_greek(c) for c in latex_notation.loc[idx[:,:,'scrap_correction']].index.get_level_values('consumer_type').to_list()
    ]

# car type 
price_greek = lambda j: f'$\\theta_{j}^0$'
latex_notation.loc[idx[:,:,'car_type', :], 'greek'] = [
    price_greek(c) for c in latex_notation.loc[idx[:,:,'car_type']].index.get_level_values('car_type').to_list()
    ]

# car type age
price_greek = lambda j: f'$\\theta_a_{j}^a$'
latex_notation.loc[idx[:,:,'car_type_age', :], 'greek'] = [
    price_greek(c) for c in latex_notation.loc[idx[:,:,'car_type_age']].index.get_level_values('car_type').to_list()
    ]

# buying
price_greek = lambda c: f'$\\rho_{c}$'
latex_notation.loc[idx[:,:,'buying', :], 'greek'] = [
    price_greek(c) for c in latex_notation.loc[idx[:,:,'buying']].index.get_level_values('consumer_type').to_list()
    ]

# adding greeks 
tab=pd.concat([tab, latex_notation], axis=1)
tab= tab[['greek', 'true value', 'mean', 'MCSE', 'MASE']]
breakpoint()
# storing as latex
with open(f'./output/simulations/output_tables/mc_table.tex', 'w') as f: 
    f.write(tab.style.format(precision=3).to_latex())
