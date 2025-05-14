function [V_new, policy_K] = bellman_2(k_grid, z_grid, V_next, prob_z_transition, params, cost_type)

    Nk = params.k_points;
    Nz = params.logz_points;

    % =====================================================================
    % Expectation and potential profits ===================================
    % =====================================================================
    profit_noadjust = z_grid' .* (k_grid.^params.theta) - params.R .* k_grid; % Dim: (Nk x Nz)
    k_next_noadjust = (1 - params.delta) * k_grid; % Dim: (Nk x 1)

    % Interpolate V_next_guess to get continuation values
    V_interp_from_k_grid = zeros(Nk, Nz); % (Nk x Nz_next)
    for i_z_next = 1:params.logz_points
        V_interp_from_k_grid(:, i_z_next) = interp1(k_grid, V_next(:, i_z_next), k_next_noadjust, 'linear', 'extrap');
    end
    EV = V_interp_from_k_grid * prob_z_transition'; % Dim: (Nk x Nz)


    % =====================================================================
    % NOT ADJUST ==========================================================
    % =====================================================================
    V_no_adjust = profit_noadjust  + params.beta * EV;


    % =====================================================================
    % ADJUST ==============================================================
    % =====================================================================   
    profit_adjust = reshape(profit_noadjust, [1, Nk, Nz]); % Dim: (1 x Nk_prime x Nz)
    % Expectation calculation
    V_future = reshape(params.beta * EV, [1, Nk, Nz]);

    % Compute adjustment costs 
    adj_cost = 0.; % assumin no adjustment cost by default 
    if strcmpi(cost_type, 'fixed') % fixed cost
        adj_cost = params.F;
    elseif strcmpi(cost_type, 'proportional') 
        adj_cost_base_for_P = profit_noadjust;
        adj_cost = params.P * max(0, adj_cost_base_for_P); % (Nk_prime x Nz)
        adj_cost = reshape(adj_cost, [1, Nk, Nz]); % Dim: (1 x Nk_prime x Nz)
    end

    V_adjust = profit_adjust - adj_cost + V_future; % Dim (Nk x Nk_prime x Nz)
    [max_value_if_adjust, idx_best_k_prime] = max(V_adjust, [], 2); 
    V_adjust = squeeze(max_value_if_adjust); 
    policy_K_adjust = k_grid(squeeze(idx_best_k_prime));

 
    % =====================================================================
    % COMBINE AND FINAL POLICY ============================================
    % =====================================================================   
    V_adjust_expanded = repmat(V_adjust', Nk, 1); % Dim: (Nk x Nz)
    V_new = max(V_no_adjust, V_adjust_expanded);
    adjust_decision = (V_adjust_expanded > V_no_adjust); % (Nk x Nz)
    
    policy_K_if_adjusts = repmat(policy_K_adjust', Nk, 1); % (Nk, Nz)
    
    current_k_mat = repmat(k_grid, 1, Nz); 
    policy_K = adjust_decision .* policy_K_if_adjusts + (~adjust_decision) .* current_k_mat;

end
