function [k_star_optimized_z, k_lower_band, k_upper_band] = extract_policy_bands(k_grid, adjust_decision_matrix, k_star_target_for_each_z)
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