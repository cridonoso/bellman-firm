function [V_new, policy_K, adjust_decision] = bellman_1(...
    k_grid, z_grid, V_next, prob_z_transition, ...
    params, cost_type)
    % k_grid = (300x1)
    % z_grid = (12x1)
    % V_next = (300x12)
    % prob_z_transition (12x12)
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
    % Expectation calculation
    V_future = reshape(params.beta * E_V, [1, Nk, Nz]); % Dim: (1 x Nk_prime x Nz)
    
    % Compute adjustment costs 
    if strcmpi(cost_type, 'fixed') % fixed cost
        profit_if_adjust = profit_if_not_adjust - params.F;
        profit_curr_rshp = reshape(profit_if_adjust, [Nk, 1, Nz]); % (Nk_actual x 1 x Nz)
    elseif strcmpi(cost_type, 'proportional') 
        profit_if_adjust = (1 - params.P) * profit_if_not_adjust; % (Nk x Nz)
        profit_curr_rshp = reshape(profit_if_adjust, [Nk, 1, Nz]); % (Nk_actual x 1 x Nz)
    end

    % Final value function and policy
    V_new =  profit_curr_rshp + V_future; % Dim (Nk x Nk_prime x Nz)
    [V_new, idx_best_k] = max(V_new, [], 2);

    V_adjust = squeeze(V_new);
    policy_K_adjust = k_grid(squeeze(idx_best_k));

    % Combine
    V_new = max(V_no_adjust, V_adjust);
    adjust_decision = (V_adjust > V_no_adjust);
    current_k= repmat(k_grid, 1, Nz);
    policy_K = adjust_decision .* policy_K_adjust + (~adjust_decision) .* current_k;
end

