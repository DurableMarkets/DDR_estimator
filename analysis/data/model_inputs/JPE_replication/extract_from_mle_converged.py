import scipy.io

# Load the .mat file
mat_data = scipy.io.loadmat('analysis/data/model_inputs/JPE_replication/mle_converged.mat')

# Now mat_data is a dictionary containing the variables from the .mat file
print(mat_data.keys())
sol_mle = mat_data['sol_mle']
mp_mle = mat_data['mp_mle']
Avar_mle = mat_data['Avar_mle']

# prices
prices=sol_mle[0][0][0]

# scrap probs: 
sol_mle[0][0][2][0][0][2]



breakpoint()