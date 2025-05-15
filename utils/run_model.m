function run_model(params)
    addpath("utils");

    % Discretization
    k_grid = linspace(params.k_min, params.k_max, params.k_points);
    k_grid = k_grid(:); % Convert to column vector (N_k x 1)

    [logz_grid, prob_z_transition] = tauchen_hussey(params.logz_points, ...
                                        params.rho, params.sigma, params.mu);
    z_grid = exp(logz_grid);

    V_guess = zeros(params.k_points, params.logz_points);

    for i = 1:length(params.scenario)
        for j= 1:length(params.cost_type)
            sc = params.scenario(i);
            ct = params.cost_type{j};
            for iter = 1:params.niter
                V_next = V_guess;

                if sc == 1
                    [V_new, policy_K, decision] = bellman_1(k_grid, z_grid, ...
                        V_next, prob_z_transition, params, ct);
                elseif sc == 2
                    [V_new, policy_K, decision] = bellman_2(k_grid, z_grid, ...
                        V_next, prob_z_transition, params, ct);
                else
                    disp('Model not defined');
                end

                [stop, V_guess] = check_convergence(V_new, V_next, iter, params);
                if stop, break, end
            end

            [k_star, k_lower, k_upper] = extract_policy_bands(k_grid, decision, policy_K);
            
            save_results(V_guess, k_star, k_lower, k_upper, z_grid, params.exp_name, sc, ct);
        end
    end
end