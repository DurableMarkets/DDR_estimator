
def create_A_B_E_matrices(X, ccps, feasible_idx, model_struct_arrays, model_funcs, params, options):
    E = - np.log(ccps)
    F = create_iota_space(feasible_idx, model_struct_arrays, model_funcs, params, options)
    
    Fu = self.model.statetransition_unconditional(F, ccps)
    denom = np.eye(self.model.n) - self.model.beta * Fu # (n, n)
    this_u = (ccps[:,:,None] * X).sum(axis=1) # (n, K) [ccps is broadcast over 3rd dimension (k, regressors)]
    this_E = (ccps * E).sum(axis=1) # (n, )

    A = np.linalg.solve(denom, this_u) # (n, )
    
    B = np.linalg.solve(denom, this_E) # (n, )

    return A, B

def create_miessi_regressors(self, A, B):
    # Create flow utility regressors
    X=self.create_regressors()
    Yoda  = self.create_F_space()
    Xm=X + Yoda @ A
    
    return Xm
