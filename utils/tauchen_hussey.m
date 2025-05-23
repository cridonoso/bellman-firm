function [actualPoints, probabilities] = tauchen_hussey(n, rho, sigma, mu)
% TAUCHEN_HUSSEY Discretizes a continuous AR(1) process into a finite-state
%                Markov chain using a method similar to Tauchen (1986) and
%                Kopecky & Suen (2010).
%
%   The AR(1) process is defined as:
%       y_t = (1 - rho) * mu + rho * y_{t-1} + epsilon_t
%   where epsilon_t is normally distributed with mean 0 and variance sigma^2,
%   i.e., epsilon_t ~ N(0, sigma^2).
%
%   The function generates 'n' grid points for the state y and an (n x n)
%   transition probability matrix.
%
%   Args:
%       n (int): The number of discrete states (grid points) for y.
%                Must be >= 2.
%       rho (double): The autoregressive coefficient of the AR(1) process.
%                     Should be in (-1, 1) for stationarity, though the code
%                     checks for abs(rho) > 1.
%       sigma (double): The standard deviation of the innovation term epsilon_t.
%                       Must be non-negative.
%       mu (double, optional): The unconditional mean of the AR(1) process y_t.
%                              Defaults to 0.0 if not provided.
%
%   Returns:
%       actualPoints (double column vector): A vector of 'n' grid points
%                                            representing the discrete states for y,
%                                            sorted in ascending order (n x 1).
%                                            These points are derived from
%                                            Gauss-Hermite quadrature nodes scaled
%                                            for a N(mu, sigma^2) distribution.
%       probabilities (double matrix): The (n x n) transition probability matrix.
%                                      probabilities(i,j) is the probability of
%                                      transitioning from state actualPoints(i)
%                                      to state actualPoints(j), i.e., P(y_t = actualPoints(j) | y_{t-1} = actualPoints(i)).
%                                      Each row sums to 1.
%
%   Dependencies:
%       - gaussnorm.m: This function is called to generate initial grid points
%                      and weights based on Gaussian quadrature for a normal
%                      distribution.
%
%   Note on the Method:
%       The grid points are generated using Gauss-Hermite quadrature nodes,
%       scaled to match a normal distribution N(mu, baseSigma^2), where
%       'baseSigma' is set to 'sigma' (std. dev. of innovations) in this
%       implementation. The transition probabilities are constructed based on a
%       ratio of conditional to unconditional normal probability densities,
%       weighted by the quadrature weights, similar to Kopecky and Suen (2010).

    % Set default for mu if not provided
    if nargin < 4
        mu = 0.0;
    end

    % Input validation
    if n < 2
        error('tauchen_hussey:NumberOfNodes', 'Only intended for at least n >= 2 nodes.');
    elseif rho < -1 || rho > 1 % Allows for rho = +/- 1. For stationarity, abs(rho) < 1 is typical.
        error('tauchen_hussey:Stationarity', 'The AR(1) coefficient rho should satisfy abs(rho) <= 1. For strict stationarity, abs(rho) < 1.');
    elseif sigma < 0
        error('tauchen_hussey:NegativeSigma', 'Negative standard deviation (sigma) provided.');
    end

    % Determine the standard deviation for generating initial grid points.
    % In this version, baseSigma is the std. dev. of the AR(1) shock (epsilon_t).
    % The commented-out part would use the unconditional std. dev. of y_t if rho < 1.
    baseSigma = sigma; % / sqrt(1 - rho^2); % User's code has this commented out.

    % Generate initial grid points (actualPoints) and quadrature weights (weights)
    % based on a normal distribution N(mu, baseSigma^2).
    [actualPoints, weights] = gaussnorm(n, mu, baseSigma^2); % weights(j) is for actualPoints(j)

    % Initialize the transition matrix
    transition = zeros(n, n);

    % Construct the (unnormalized) transition matrix elements
    % transition(i,j) will be related to P(next_state = actualPoints(j) | current_state = actualPoints(i))
    for i = 1:n % Loop over current states y_i = actualPoints(i)
        for j = 1:n % Loop over next states y_j = actualPoints(j)
            % ymean is the shock epsilon_t needed to go from E[y_t|y_{t-1}=y_i] to y_j
            % E[y_t|y_{t-1}=y_i] = (1-rho)*mu + rho*y_i
            % So, ymean = y_j - ((1-rho)*mu + rho*y_i)
            ymean_shock = actualPoints(j) - rho * actualPoints(i) - (1 - rho) * mu;
            
            % Probability density of this shock occurring (conditional density part)
            transProb = normpdf(ymean_shock, 0, sigma); % PDF of N(0, sigma^2)
            
            % Deviation of the next state y_j from the unconditional mean mu
            y_deviation_from_mean = actualPoints(j) - mu;
            
            % "Unconditional" probability density part, based on N(mu, baseSigma^2)
            % which is the distribution from which actualPoints were effectively drawn.
            uncondProb = normpdf(y_deviation_from_mean, 0, baseSigma);
            
            % Formula for transition matrix element (Kopecky & Suen, 2010, style)
            % Ensure uncondProb is not zero to avoid division by zero.
            if uncondProb > 1e-100 % A small threshold to prevent NaN/Inf
                transition(i, j) = weights(j) * (transProb / uncondProb);
            else
                % If uncondProb is ~0, it implies actualPoints(j) is very far in the tails
                % of N(mu, baseSigma^2). If transProb is also ~0, then 0.
                % If transProb is not ~0, this would be a large number.
                % This case should be rare if n is reasonably large and grid covers well.
                transition(i, j) = 0; % Or handle as per specific theory if needed
            end
        end
    end

    % Normalize each row of the transition matrix to sum to 1, forming probabilities
    % probabilities(i,j) = P(y_t = actualPoints(j) | y_{t-1} = actualPoints(i))
    row_sums = sum(transition, 2);
    
    % Handle cases where a row sum might be zero (e.g., if all transProb/uncondProb were zero for a row)
    % by distributing probability масса uniformly or to a self-loop, or erroring.
    % Here, if a row sums to 0, it will result in NaNs. A common fix is to assign 1 to a diagonal.
    if any(row_sums == 0)
        warning('tauchen_hussey:ZeroRowSum', 'Some rows in the transition matrix summed to zero before normalization. Resulting probabilities for these rows will be NaN or Inf. Consider checking parameters or grid density.');
        % Example: replace NaN rows with uniform or self-loop
        % For rows with sum 0, make it a self-loop to avoid NaNs, or uniform
        % zero_sum_rows = find(row_sums == 0);
        % for r_idx = zero_sum_rows'
        %     transition(r_idx, r_idx) = 1; % Self-loop
        %     row_sums(r_idx) = 1;
        % end
    end
    
    probabilities = transition ./ row_sums; % Element-wise division for each row
    
end