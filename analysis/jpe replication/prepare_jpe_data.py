import jpe_replication.process_data.choice as choice
import jpe_replication.process_data.scrap as scrap
import numpy as np
#import jpe_replication.process_data.prices as prices
# Options:
kwargs = {
    'indir': './analysis/data/8x4/',
    'outdir': './analysis/data/8x4_eqb/',
    'years': np.arange(1996, 2009).astype(str).tolist(),
    'max_age_car': 22,
}

# Create choice data 
choice.process_choice_data(**kwargs)

# Create scrap data 
scrap.process_scrap_data(**kwargs)

