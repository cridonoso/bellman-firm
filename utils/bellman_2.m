function [V_new_matrix, policy_K_matrix] = bellman_2(k_grid, z_grid_col, V_next_guess, prob_z_transition, params, cost_type)
    % SOLVE_BELLMAN_CASE_II Computes one iteration of the value function for Case (ii)
    % ("ready-to-use" capital) with either fixed (F) or proportional (P) adjustment costs.
    % The function is fully vectorized.
    %
    % Args:
    %   k_grid: Column vector (Nk x 1) of capital grid points.
    %   z_grid_col: Column vector (Nz x 1) of productivity grid points.
    %   V_next_guess: Matrix (Nk x Nz) with V(K', z') from the next period/iteration.
    %   prob_z_transition: Transition matrix for z (Nz x Nz), P(z_j | z_i).
    %   params: Struct with model parameters (beta, R, theta, delta, F, P, etc.).
    %   cost_type: String, 'fixed' for cost F, 'proportional' for cost P.
    %
    % Returns:
    %   V_new_matrix: Matrix (Nk x Nz) with the updated value function V(K, z).
    %   policy_K_matrix: Matrix (Nk x Nz) with the optimal capital policy K'.

    % Ensure z_grid is a row vector for broadcasting
    z_grid_row = z_grid_col'; % If z_grid_col is (Nz x 1), z_grid_row is (1 x Nz)

    % Extract model parameters for readability
    beta = params.beta;
    R_val = params.R; % Renamed to avoid conflict if R is a MATLAB function
    theta = params.theta;
    delta = params.delta;
    F_cost = params.F; % Fixed cost F
    P_cost = params.P; % Proportional cost fraction P

    % --- A. "No Adjustment" Branch ---
    % k_grid here represents k_in (capital at the beginning of the period)

    % Profit if no adjustment is made, for all (k_in, z) combinations
    profit_if_not_adjust = z_grid_row .* (k_grid.^theta) - R_val .* k_grid; % (Nk x Nz)

    % Next period’s capital if no adjustment: (1 - delta) * k_in
    K_next_if_not_adjust_vec = (1 - delta) * k_grid; % (Nk x 1)

    % Interpolate V_next_guess to get continuation values
    V_interp_for_no_adjust = zeros(params.k_points, params.logz_points); % (Nk x Nz_next)
    for i_z_next = 1:params.logz_points
        V_interp_for_no_adjust(:, i_z_next) = interp1(k_grid, V_next_guess(:, i_z_next), K_next_if_not_adjust_vec, 'linear', 'extrap');
    end

    % Expected continuation value if no adjustment: E[V((1-delta)k_in, z') | z]
    E_V_no_adjust = V_interp_for_no_adjust * prob_z_transition'; % (Nk x Nz)

    % Total value of the "No Adjustment" branch
    V_no_adjust_branch = profit_if_not_adjust + beta * E_V_no_adjust; % (Nk x Nz)

    % --- B. "Adjustment" Branch ---
    % K_prime_candidates are the possible capital levels K' to which the firm can adjust
    K_prime_candidates = k_grid; % (Nk_prime x 1), where Nk_prime = params.k_points

    % Profit if adjusting to K', for all (K', z) combinations.
    % This is: z * (K')^theta - R_val * K'
    Profit_at_K_prime_z = z_grid_row .* (K_prime_candidates.^theta) - R_val .* K_prime_candidates; % (Nk_prime x Nz)

    % Compute specific adjustment costs
    Adjustment_cost_at_K_prime_z = zeros(params.k_points, params.logz_points); % (Nk_prime x Nz)
    if strcmpi(cost_type, 'fixed')
        Adjustment_cost_at_K_prime_z = F_cost; % Fixed cost F, will broadcast
    elseif strcmpi(cost_type, 'proportional')
        % Cost is P * max(0, profit with K')
        Adjustment_cost_at_K_prime_z = P_cost * max(0, Profit_at_K_prime_z);
    else
        error('Unrecognized cost type in solve_bellman_case_ii. Use "fixed" or "proportional".');
    end

    % Next period’s capital if adjustment is made to K': (1 - delta) * K'
    K_prime_next_if_adjust_vec = (1 - delta) * K_prime_candidates; % (Nk_prime x 1)

    % Interpolate V_next_guess to get continuation values for adjustment
    V_interp_for_adjust = zeros(params.k_points, params.logz_points); % (Nk_prime x Nz_next)
    for i_z_next = 1:params.logz_points
        V_interp_for_adjust(:, i_z_next) = interp1(k_grid, V_next_guess(:, i_z_next), K_prime_next_if_adjust_vec, 'linear', 'extrap');
    end

    % Expected continuation value if adjusting to K': E[V((1-delta)K', z') | z]
    E_V_adjust_k_prime_z = V_interp_for_adjust * prob_z_transition'; % (Nk_prime x Nz)

    % Total value if adjusting to K' (before maximization over K')
    % That is: z(K')^theta - R*K' - AdjustmentCost + beta * E[V((1-delta)K',z')]
    Value_candidate_if_adjust = Profit_at_K_prime_z - Adjustment_cost_at_K_prime_z + beta * E_V_adjust_k_prime_z; % (Nk_prime x Nz)

    % Maximize over K' for each productivity level z
    % V_adjust_optimal_for_z_row(iz) is max_{K'} {value of adjusting to K' given z_grid_col(iz)}
    [V_adjust_optimal_for_z_row, Idx_optimal_K_prime_for_z_row] = max(Value_candidate_if_adjust, [], 1); % Max over dim 1 (K_prime_candidates)
                                                                                                          % Result is (1 x Nz)
    % Optimal K' policy if adjustment is chosen, for each z
    Policy_K_prime_chosen_for_z_col = k_grid(Idx_optimal_K_prime_for_z_row'); % (Nz x 1)

    % Value of the "Adjustment" branch (after optimally choosing K'), only depends on z.
    % Expand to compare with V_no_adjust_branch across all k_in.
    V_adjust_branch = repmat(V_adjust_optimal_for_z_row, params.k_points, 1); % (Nk x Nz)

    % --- C. Combine: Max over "No Adjustment" and "Adjustment" ---
    V_new_matrix = max(V_no_adjust_branch, V_adjust_branch); % (Nk x Nz)

    % --- D. Determine the Optimal Policy Function ---
    % Boolean matrix indicating whether to adjust or not
    Adjust_decision_mat = (V_adjust_branch > V_no_adjust_branch); % (Nk x Nz)

    % Construct the policy matrix for K'
    % If adjusting, K' is the optimal one for that z. If not, K' = current k_in.
    Policy_K_if_adjust_mat = repmat(Policy_K_prime_chosen_for_z_col', params.k_points, 1); % (Nk x Nz)
    Current_K_mat = repmat(k_grid, 1, params.logz_points); % Matrix of current k_in (Nk x Nz)

    policy_K_matrix = Adjust_decision_mat .* Policy_K_if_adjust_mat + (~Adjust_decision_mat) .* Current_K_mat;
end
