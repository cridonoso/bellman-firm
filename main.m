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

[logz_grid, prob_z_transition] = tauchen_hussey(params.logz_points, params.rho, params.sigma, params.mu);
z_grid = exp(logz_grid); % Inverse log transformation

% Initial guess for the value function (i.e., zeros)
V_guess = zeros(params.k_points, params.logz_points);

% To store the partial policy values
policy_K_matrix = zeros(params.k_points, params.logz_points);

tic; % Start timer
for iter = 1:params.niter
    V_next_iter = V_guess;
    
    [V_new_iter_1, policy_K_matrix_iter_1] = bellman_1(k_grid, z_grid, ...
        V_next_iter, prob_z_transition, params, 'fixed');

    [V_new_iter_2, policy_K_matrix_iter_2] = bellman_2(k_grid, z_grid, ...       
        V_next_iter, prob_z_transition, params, 'fixed');

    % Check convergence
    diff = max(abs(V_new_iter_2(:) - V_guess(:)));
    if diff < params.tol
        disp(['Convergence achieved at iteration: ', num2str(iter), ', Diferencia: ', num2str(diff)]);
        V_guess = V_new_iter_2;               % Save the best value
        policy_K_matrix = policy_K_matrix_iter_2; % Save the final policy
        break; 
    end
    
    % Update guess for the next iteration
    V_guess = V_new_iter_2;
    if iter == params.niter
        disp('Maximum number of iterations reached. No convergence achieved.');
        policy_K_matrix = policy_K_matrix_iter_2;
    end
end
toc; % Stop timer
tic; % Start timer again
