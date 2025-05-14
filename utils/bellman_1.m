function [V_new, policy_K] = bellman_1(...
    k_grid, z_grid, V_next, prob_z_transition, ...
    params, cost_type)

    Nk = params.k_points;
    Nz = params.logz_points;

    % =====================================================================
    % NOT ADJUST ==========================================================
    % =====================================================================
    % Profit if no adjustment is made, for all (k_in, z) combinations
    profit_if_not_adjust = z_grid' .* (k_grid.^params.theta) - params.R .* k_grid; 
    k_next = (1 - params.delta) * k_grid; % (Nk x 1)
    
    % Interpolate V_next_guess to get continuation values
    V_interp_for_no_adjust = zeros(Nk, Nz); % (Nk x Nz_next)
    for i_z_next = 1:params.logz_points
        V_interp_for_no_adjust(:, i_z_next) = interp1(k_grid, V_next(:, i_z_next), k_next, 'linear', 'extrap');
    end
    % Expected continuation value if no adjustment: E[V((1-delta)k_in, z') | z]
    E_V = V_interp_for_no_adjust * prob_z_transition'; % (Nk x Nz)
    V_no_adjust = profit_if_not_adjust + params.beta * E_V; % (Nk x Nz)


    % =====================================================================
    % ADJUST ==============================================================
    % =====================================================================   
    profit_curr_rshp = reshape(profit_if_not_adjust, [Nk, 1, Nz]); % Dim: (Nk x 1 x Nz)
    % Expectation calculation
    V_future = reshape(params.beta * E_V, [1, Nk, Nz]); % Dim: (1 x Nk_prime x Nz)
    
    % Compute adjustment costs 
    adj_cost = 0.; % assumin no adjustment cost by default 
    if strcmpi(cost_type, 'fixed') % fixed cost
        adj_cost = params.F;
    elseif strcmpi(cost_type, 'proportional') 
        term_zk_theta = z_grid' .* (k_grid.^params.theta); % (Nk x Nz)
        term_Rk_prime = params.R .* k_grid; % (Nk_prime x 1)
        term_zk_theta_reshaped = reshape(term_zk_theta, [Nk, 1, Nz]);
        term_Rk_prime_reshaped = reshape(term_Rk_prime, [1, Nk, 1]);
        B_tensor = term_zk_theta_reshaped - term_Rk_prime_reshaped; % Dim: (Nk x Nk_prime x Nz)
        adj_cost = params.P * max(0, B_tensor); % Dim: (Nk x Nk_prime x Nz)
    end

    % Final value function and policy
    V_new = profit_curr_rshp - adj_cost + V_future; % Dim (Nk x Nk_prime x Nz)
    [V_new, idx_best_k] = max(V_new, [], 2);
    V_adjust = squeeze(V_new);
    policy_K_adjust = k_grid(squeeze(idx_best_k));

    % Combine
    V_new = max(V_no_adjust, V_adjust);
    adjust_decision = (V_adjust > V_no_adjust);
    current_k= repmat(k_grid, 1, Nz);
    policy_K = adjust_decision .* policy_K_adjust + (~adjust_decision) .* current_k;
end

