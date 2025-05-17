function run_model(params)
% RUN_MODEL Solves a dynamic firm investment model by iterating over the
%           Bellman equation for different scenarios and adjustment cost types.
%
%   This function serves as the main driver for solving a firm's dynamic
%   optimization problem. It sets up the state space (capital and productivity
%   grids), then iterates the value function until convergence for each
%   combination of model scenarios (e.g., different Bellman equation solvers
%   like bellman_1 or bellman_2) and adjustment cost types (e.g., fixed or
%   proportional) specified in the 'params' structure.
%
%   After convergence for a given configuration, it extracts policy functions
%   (optimal capital k*, inaction bands k_lower and k_upper) and saves the
%   results to a .mat file.
%
%   Args:
%       params (struct): A structure containing all necessary parameters for
%                        the model, discretization, and solution process.
%                        Key fields expected:
%           - k_min (double): Minimum value for the capital grid.
%           - k_max (double): Maximum value for the capital grid.
%           - k_points (int): Number of points in the capital grid.
%           - logz_points (int): Number of points for the productivity grid.
%           - rho (double): Autocorrelation coefficient for the log(z) process.
%           - sigma (double): Standard deviation of the shock to log(z) process.
%           - mu (double): Mean of the log(z) process (often 0 for demeaned).
%           - scenario (vector): A list of scenario identifiers (e.g., [1, 2])
%                                that determine which Bellman solver to use.
%           - cost_type (cell array): A list of strings specifying adjustment
%                                     cost types (e.g., {'fixed', 'proportional'}).
%           - niter (int): Maximum number of iterations for value function convergence.
%           - exp_name (char array): Experiment name, used for saving results.
%           % (Other parameters like theta, R, delta, beta, F, P are assumed to be
%           %  fields in 'params' and are passed to the Bellman functions).
%
%   Function Dependencies:
%       - addpath("utils"): Assumes utility functions are in a 'utils' subdirectory.
%       - tauchen_hussey.m: Used to discretize the AR(1) productivity process.
%       - bellman_1.m: Solves the Bellman equation for scenario 1.
%       - bellman_2.m: Solves the Bellman equation for scenario 2.
%       - check_convergence.m: Checks for value function convergence and updates guess.
%       - extract_policy_bands.m: Extracts k*, k_lower, k_upper from converged policies.
%       - save_results.m: Saves the results to a .mat file.
%
%   Side Effects:
%       - Adds 'utils' to the MATLAB path.
%       - Prints convergence status and differences to the command window.
%       - Saves .mat files containing results for each scenario and cost type.

    addpath("utils"); % Adds the 'utils' directory to the MATLAB path

    % --- 1. Discretize State Space ---
    % Capital grid (k_grid)
    k_grid = linspace(params.k_min, params.k_max, params.k_points);
    k_grid = k_grid(:); % Ensure k_grid is a column vector (Nk x 1)

    % Productivity grid (z_grid) and transition matrix (prob_z_transition)
    % tauchen_hussey discretizes an AR(1) process for log(z).
    [logz_grid, prob_z_transition] = tauchen_hussey(params.logz_points, ...
                                            params.rho, params.sigma, params.mu);
    z_grid = exp(logz_grid); % Convert log(z) grid to levels
    
    % Initial guess for the value function V(k,z)
    V_guess = zeros(params.k_points, params.logz_points);

    % --- 2. Iterate over Scenarios and Cost Types ---
    for i = 1:length(params.scenario) % Loop through specified scenarios
        for j = 1:length(params.cost_type) % Loop through specified cost types
            sc = params.scenario(i);       % Current scenario identifier
            ct = params.cost_type{j};      % Current cost type string
            
            diff_per_iter = zeros(params.niter, 1); % Store convergence differences for this run

            % --- 3. Value Function Iteration ---
            for iter = 1:params.niter
                V_next = V_guess; % Use the previous iteration's V as the guess for V_t+1

                % Select Bellman equation solver based on scenario
                if sc == 1
                    [V_new, policy_K, decision] = bellman_1(k_grid, z_grid, ...
                                V_next, prob_z_transition, params, ct);
                elseif sc == 2
                    [V_new, policy_K, decision] = bellman_2(k_grid, z_grid, ...
                                V_next, prob_z_transition, params, ct);
                else
                    disp('Error: Model (scenario) not defined.');
                    return; % Exit if scenario is not recognized
                end

                % Check for convergence
                [stop, V_guess, diff] = check_convergence(V_new, V_next, iter, params);
                diff_per_iter(iter) = diff; % Store difference
                
                if stop % If converged
                    diff_per_iter = diff_per_iter(1:iter); % Trim unused part of diff_per_iter
                    break; 
                end
            end % End of value function iteration loop

            % --- 4. Extract Policy Bands and Save Results ---
            % extract_policy_bands uses the converged adjustment decision and policy_K
            % (which might be k* target if always adjusting, or k_t+1 choices)
            % to determine the inaction region [k_lower, k_upper] and the k* target.
            % Note: The 'policy_K' passed here is k_t+1(k_t, z_t).
            % 'k_star_target_for_each_z' in extract_policy_bands usually refers to the
            % capital level targeted when adjustment occurs. This might be derived from policy_K
            % or be a separate output from the Bellman, depending on the Bellman function's design.
            % Here, policy_K itself (which contains k_t+1 choices) is passed as the target.
            [k_star, k_lower, k_upper] = extract_policy_bands(k_grid, decision, policy_K);
            
            % Save all relevant results for this scenario and cost type
            save_results(V_guess, k_star, k_lower, k_upper, z_grid, params.exp_name, sc, ct, diff_per_iter);
            
        end % End of cost type loop
    end % End of scenario loop
end