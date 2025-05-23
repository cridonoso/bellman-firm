% ------------------------------------------------------------------------
% Author: Cristobal Donoso
% Affiliation: PhD Student in Economics, Universidad de Chile
% Date: May 16, 2025
%
% Description:
% This script runs a dynamic firm investment model under different
% parameterizations, specifically varying sigma (volatility) and delta
% (depreciation/loss fraction) for different problem sets (p3, p7, p8).
% It loads base parameters from a JSON file, then iterates through
% specified ranges of sigma and/or delta, calling the 'run_model'
% function for each configuration and saving the results in structured
% subfolders.
% ------------------------------------------------------------------------
clearvars
clc

addpath("utils"); % Add utility functions to path

% Load base parameters from a JSON configuration file
% Assumes params_p3.json is a representative file for loading initial common params
% Modify 'params_p3.json' if a different base configuration is needed.
str = fileread('./config/params_p0.json'); 
params_base = jsondecode(str);

tic; % Start timer

% Determine execution path based on experiment name (exp_name) in params
if isscalar(params_base.sigma) && isscalar(params_base.delta) 
    % Standard run with a single sigma value (e.g., for p3, p4, p5, p6 type problems)
    disp(['Running single configuration for experiment: ', char(params_base.exp_name)])
    target_folder = sprintf('./backup/%s/', char(params_base.exp_name));
    params_base.exp_name = target_folder; % Set new path for saving results
    run_model(params_base);      % Run the model with these parameters
    
elseif strcmp(params_base.exp_name, "p7") % Check string equality
    % Special handling for problem p7: iterate over a vector of sigmas
    disp('Running configurations for Problem 7 (varying sigma)')
    if ~isfield(params_base, 'sigma') || ~isvector(params_base.sigma)
        error('Error: params_base.sigma must be a vector for p7.');
    end
    for i = 1:length(params_base.sigma)
        params = params_base;  % Create a mutable copy of base parameters
        params.sigma = params_base.sigma(i);  % Set current sigma
        
        % Define target folder for results of this specific sigma run
        target_folder = sprintf('./backup/%s/sigma_%.3f/', char(params_base.exp_name), params.sigma);

        params.exp_name = target_folder; % Pass specific folder for this run's results
                                         % (original params.exp_name kept for base ID if needed)
        
        disp(['Running p7 for sigma = ', num2str(params.sigma)]);
        run_model(params);      % Run model for current sigma
    end
    
elseif strcmp(params_base.exp_name, "p8") % Check string equality
    % Special handling for problem p8: iterate over sigmas and deltas
    disp('Running configurations for Problem 8 (varying sigma and delta)')
    if ~isfield(params_base, 'sigma') || ~isvector(params_base.sigma) || ...
       ~isfield(params_base, 'delta') || ~isvector(params_base.delta)
        error('Error: params_base.sigma and params_base.delta must be vectors for p8.');
    end
    
    for i = 1:length(params_base.sigma)
        for j = 1:length(params_base.delta)
            params = params_base;  % Create a mutable copy
            params.sigma = params_base.sigma(i);  % Set current sigma
            params.delta = params_base.delta(j);  % Set current delta
            
            % Define target folder for results of this specific (sigma, delta) run
            target_folder = sprintf('./backup/%s/sigma_%.2f_delta_%.2f/', ...
                                    char(params_base.exp_name), params.sigma, params.delta);
            params.exp_name_run = target_folder; % Pass specific folder for this run
            
            disp(['Running p8 for sigma = ', num2str(params.sigma), ', delta = ', num2str(params.delta)]);
            run_model(params);      % Run model for current (sigma, delta)
        end
    end
else
    disp(['Warning: exp_name "', char(params_base.exp_name), '" not recognized for special loop logic. Assuming scalar sigma if no error.'])
    % Fallback to scalar run if exp_name is not p7 or p8 but sigma is scalar
    if isscalar(params_base.sigma)
        target_folder = sprintf('./backup/%s/', char(params_base.exp_name));
        params_base.exp_name = target_folder; 
        run_model(params_base);
    else
        disp('Error: exp_name not p7 or p8, and sigma is not scalar. Unsure how to proceed.');
    end
end

toc; % Stop timer and display elapsed time
disp('All specified model runs completed.');
