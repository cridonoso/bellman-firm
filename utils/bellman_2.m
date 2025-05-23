function [V_new, policy_K, adjust_decision] = bellman_2(k_grid, z_grid, V_next, prob_z_transition, params, cost_type)
% BELLMAN_2 Performs one iteration of the Bellman equation for a dynamic
%           firm investment model with capital adjustment costs. (Alternative version)
%
%   This function computes the updated value function (V_new) and the
%   optimal capital policy (policy_K) for each state (k,z), given the
%   value function from the next period (V_next). It compares the value of
%   not adjusting capital versus adjusting capital.
%   NOTE: This version has a different structure for calculating the value
%   of adjustment compared to typical formulations, particularly in how
%   current profits and future values are combined before maximization.
%
%   Args:
%       k_grid (double vector): Grid of current capital stock levels (Nk x 1).
%       z_grid (double vector): Grid of current productivity shock levels (Nz x 1).
%       V_next (double matrix): Value function from the next period, V_t+1(k,z) (Nk x Nz).
%       prob_z_transition (double matrix): Transition probability matrix for z.
%                                          P(z_j_next | z_i_current) (Nz x Nz).
%                                          (Assumed: element (r,c) is P(z_next=z_r | z_current=z_c))
%       params (struct): Structure containing model parameters:
%           - theta (double): Capital share in production function.
%           - R (double): Gross rental rate or user cost of capital.
%           - delta (double): Capital depreciation rate.
%           - beta (double): Discount factor.
%           - F (double, optional): Fixed cost of capital adjustment.
%           - P (double, optional): Proportional cost of capital adjustment.
%           - k_points (int): Number of points in k_grid (Nk).
%           - logz_points (int): Number of points in z_grid (Nz).
%       cost_type (char array): Type of adjustment cost: 'fixed' or 'proportional'.
%
%   Returns:
%       V_new (double matrix): Updated value function V_t(k,z) (Nk x Nz).
%       policy_K (double matrix): Optimal next-period capital stock k_t+1(k_t,z_t) (Nk x Nz).
%                                 If not adjusting, k_t+1 = k_t.
%       adjust_decision (logical matrix): Decision to adjust capital (Nk x Nz).
%                                         True (1) if firm adjusts, False (0) otherwise.

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
        V_interp_from_k_grid(:, i_z_next) = interp1(k_grid, ...
            V_next(:, i_z_next), k_next_noadjust, 'linear', 'extrap');
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
    V_future = params.beta * EV;

    % Compute adjustment costs 
    if strcmpi(cost_type, 'fixed') % fixed cost
        profit_adjust = profit_noadjust - params.F;
    elseif strcmpi(cost_type, 'proportional') 
        profit_adjust = (1 - params.P) * profit_noadjust; % (Nk_prime x Nz)

    end

    V_adjust = profit_adjust + V_future; % Dim (Nk x Nk_prime x Nz)
    [max_value_if_adjust, idx_best_k_prime] = max(V_adjust, [], 1); 
    V_adjust = squeeze(max_value_if_adjust); 
    policy_K_adjust = k_grid(squeeze(idx_best_k_prime));
    
 
    % =====================================================================
    % COMBINE AND FINAL POLICY ============================================
    % =====================================================================   
    V_adjust_expanded = repmat(V_adjust, Nk, 1); % Dim: (Nk x Nz)
    V_new = max(V_no_adjust, V_adjust_expanded);
    adjust_decision = (V_adjust_expanded > V_no_adjust); % (Nk x Nz)
    
    policy_K_if_adjusts = repmat(policy_K_adjust', Nk, 1); % (Nk, Nz)
    
    current_k_mat = repmat(k_grid, 1, Nz); 
    policy_K = adjust_decision .* policy_K_if_adjusts + (~adjust_decision) .* current_k_mat;

end
