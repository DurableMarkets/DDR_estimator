
%% 
sol_mle=load('mle_converged.mat');


%% save output for later: 

% scrap probs
csvwrite('scrap_probabilities_model.csv',cell2mat(sol_mle.ccp_scrap_tau'))

% prices:
csvwrite('used_car_prices_model.csv', sol_mle.p)

% EV terms
csvwrite('ev_terms_model.csv', cell2mat(sol_mle.ev_tau'))

% estimates
save('mp_mle_model.mat', '-struct', 'mp_mle');

%
to_export = ['acc_0', 'acc_a', 'sigma_s', 'mum', 'psych_transcost', 'psych_transcost_nocar', 'tc_sale', 'tc_sale_even', 'u_0', 'u_a']
for i = 1:length(to_export)
    save('mp_mle_model.mat', '-struct', mp_mle.(to_export{i}));
end