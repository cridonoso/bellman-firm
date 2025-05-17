function [k_star_optimized_z, k_lower_band, k_upper_band] = extract_policy_bands(k_grid, adjust_decision_matrix, k_star_target_for_each_z)
% EXTRACT_POLICY_BANDS Identifies the inaction band and optimal target capital
%                      from a firm's adjustment decision matrix.
%
%   This function processes the output of a dynamic programming problem,
%   specifically the matrix indicating whether a firm adjusts its capital
%   or not at different states. It then determines the upper and lower
%   bounds of the capital levels where the firm chooses not to adjust (the
%   inaction band) for each productivity state z. It also returns the
%   target capital stock k*(z) that the firm aims for when it does adjust.
%
%   Args:
%       k_grid (double vector): Grid of possible capital stock levels (Nk x 1).
%                               Nk is the number of capital points.
%       adjust_decision_matrix (logical matrix): A matrix (Nk x Nz) where
%                                                element (i,j) is true (1) if
%                                                the firm chooses to adjust its
%                                                capital when current capital is k_grid(i)
%                                                and productivity is z_grid(j), and
%                                                false (0) otherwise.
%       k_star_target_for_each_z (double vector): The optimal target capital stock k*(z)
%                                                 that the firm chooses if it decides to adjust,
%                                                 for each productivity state z.
%                                                 Expected as (Nz x 1) or (1 x Nz).
%
%   Returns:
%       k_star_optimized_z (double vector): The target capital stock k*(z) for each
%                                           productivity state, ensured to be a
%                                           column vector (Nz x 1). This is often the
%                                           same as k_star_target_for_each_z but reshaped.
%       k_lower_band (double vector): The lower bound of the inaction band for capital,
%                                     k_lower(z), for each productivity state z (Nz x 1).
%                                     This is the lowest k at which the firm does NOT adjust.
%       k_upper_band (double vector): The upper bound of the inaction band for capital,
%                                     k_upper(z), for each productivity state z (Nz x 1).
%                                     This is the highest k at which the firm does NOT adjust.
%
%   Note:
%       If for a given productivity state z_j, the firm always adjusts (i.e.,
%       there is no inaction region for that z_j), both k_lower_band(j) and
%       k_upper_band(j) are set to k_star_optimized_z(j), effectively
%       making the inaction band a single point at k*.

    Nk = size(k_grid, 1);
    Nz = size(adjust_decision_matrix, 2);

    % Asegurar que k_star_target_for_each_z sea un vector columna para la salida
    if size(k_star_target_for_each_z, 1) == 1 % Si es un vector fila
        k_star_optimized_z = k_star_target_for_each_z';
    else
        k_star_optimized_z = k_star_target_for_each_z;
    end

    k_lower_band = NaN(Nz, 1);
    k_upper_band = NaN(Nz, 1);

    for j = 1:Nz % Iterar sobre cada estado de z (cada columna)
        
        % Decisiones de ajuste para el z_j actual
        current_z_adjust_decision = adjust_decision_matrix(:, j);
        
        % Encontrar los índices de k_grid donde NO se ajusta (la banda de inacción)
        inaction_k_indices = find(~current_z_adjust_decision);
        
        if isempty(inaction_k_indices)
            % Caso degenerado: la firma SIEMPRE ajusta para este z_j.
            % Esto podría pasar si los costos de ajuste son cero o muy bajos,
            % o la grilla de k es muy gruesa.
            % En este caso, la "banda" colapsa al punto k*.
            k_lower_band(j) = k_star_optimized_z(j);
            k_upper_band(j) = k_star_optimized_z(j);
        else
            % La banda de inacción existe.
            % El umbral inferior es el k más bajo en la región de inacción.
            k_lower_band(j) = k_grid(min(inaction_k_indices));
            
            % El umbral superior es el k más alto en la región de inacción.
            k_upper_band(j) = k_grid(max(inaction_k_indices));
        end
    end
end