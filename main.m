clearvars
clc

str = fileread('./config/params_p3.json');
params_base = jsondecode(str);

tic;
if isscalar(params_base.sigma)
    disp('scalar')
    target_folder = sprintf('./backup/%s/', char(params_base.exp_name));
    params_base.exp_name = target_folder; % set new path
    run_model(params_base);     % run model
else
    % exclusive code for problem 7
    for i = 1:length(params_base.sigma)
        params = params_base;  % copy params
        params.sigma = params_base.sigma(i);  % set sigma
        target_folder = sprintf('./backup/%s/sigma_%.2f/', char(params_base.exp_name), params.sigma);
        params.exp_name = target_folder; % set new path
        run_model(params);     % run model
    end
end
toc; % Stop timer

