function save_results(value, k_star, k_lower, k_upper, z, folder, scenario, cost, diff_per_iter)
% SAVE_RESULTS Saves the results of a model solution to a .mat file.
%
%   This function takes the converged value function, policy function components
%   (optimal capital k*, inaction band bounds k_lower, k_upper), the
%   productivity grid, and convergence history, then saves them into a .mat
%   file. The filename is constructed based on the scenario identifier and
%   cost type. If the target folder does not exist, it is created.
%
%   Args:
%       value_func (double matrix): The converged value function V(k,z).
%                                   Typically (Nk x Nz).
%       k_star (double vector or matrix): The optimal capital policy k*(z) or k*(k,z).
%                                         If it's k*(z), typically (Nz x 1).
%                                         If it's k*(k,z) from Bellman, (Nk x Nz).
%       k_lower (double vector): Lower bound of the inaction band k_lower(z) (Nz x 1).
%       k_upper (double vector): Upper bound of the inaction band k_upper(z) (Nz x 1).
%       z_grid_final (double vector): The grid used for productivity states z (Nz x 1).
%                                     (Renamed from 'z' in function signature for clarity).
%       folder_name (char array): The name of the main folder where results should be
%                                 saved (e.g., an experiment name like 'p3_results').
%                                 The function will create this folder if it doesn't exist.
%       scenario_id (double or int): Identifier for the model scenario (e.g., 1 or 2).
%       cost_type_str (char array): String describing the cost type (e.g., 'fixed',
%                                   'proportional'). (Renamed from 'cost' for clarity).
%       diff_history (double vector): A vector containing the history of differences
%                                     (e.g., max absolute difference between V_new and V_next)
%                                     during value function iteration. (N_iter x 1).
%
%   File Naming Convention:
%       The output .mat file will be named as: "[scenario_id]_[cost_type_str].mat"
%       (e.g., "1_fixed.mat") and saved within the specified 'folder_name'.
%
%   Side Effects:
%       - Creates the 'folder_name' directory if it does not already exist.
%       - Saves a .mat file containing the input arguments as variables.
%       - Displays a success message "Matrices successfully saved!" to the command window.

    fn_value  = sprintf('%d_%s.mat', scenario, cost);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    path  = fullfile(folder, fn_value);

    save(char(path), 'value', 'k_star', 'k_upper', 'k_lower', 'z', 'diff_per_iter')
    disp('Matrices successfully saved!');
end
