function save_results(value, k_star, k_lower, k_upper, z, folder, scenario, cost)
    fn_value  = sprintf('%d_%s.mat', scenario, cost);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    path  = fullfile(folder, fn_value);
    save(char(path), 'value', 'k_star', 'k_upper', 'k_lower', 'z')
    disp('Matrices successfully saved!');
end
