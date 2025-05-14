clearvars
clc
%% Question 4 
addpath("utils") % Add utility functions

% Load model hyperparameters
str = fileread('params.json');
params = jsondecode(str);

% Discretization
k_grid = linspace(params.k_min, params.k_max, params.k_points);
k_grid = k_grid(:); % Convert to column vector (N_k x 1)

[logz_grid, prob_z_transition] = tauchen_hussey(params.logz_points, ...
                                    params.rho, params.sigma, params.mu);
z_grid = exp(logz_grid); % Inverse log transformation

% Initial guess for the value function (i.e., zeros)
V_guess = zeros(params.k_points, params.logz_points);

% To store the partial policy values
policy_K_matrix = zeros(params.k_points, params.logz_points);

tic; % Start timer
for iter = 1:params.niter
    V_next = V_guess;
    
    [V_new, policy_K] = bellman_1(k_grid, z_grid, ...
        V_next, prob_z_transition, params, 'proportional');

    % Check convergence
    [stop, V_guess] = check_convergence(V_new, V_next, iter, params);
        
    if stop, break, end
end
toc; % Stop timer
tic; % Start timer again

max_k = max(policy_K, [], 1);
min_k = min(policy_K, [], 1);
mean_k = mean(policy_K, 1);
plot(z_grid, max_k);hold on
plot(z_grid, min_k);hold on
plot(z_grid, mean_k);