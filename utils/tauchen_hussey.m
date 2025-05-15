function [actualPoints, probabilities] = tauchen_hussey(n, rho, sigma, mu)
    % Tauchen-Hussey algorithm for discretizing AR(1) process:
    % y_t = (1 - rho) * mu + rho * y_{t-1} + epsilon_t, epsilon ~ N(0, sigma^2)

    if nargin < 4
        mu = 0.0;
    end

    if n < 2
        error('Only intended for at least n > 2 nodes');
    elseif rho < -1 || rho > 1
        error('The process needs to be covariance stationary');
    elseif sigma < 0
        error('Negative standard deviation provided');
    end

    baseSigma = sigma ;%/ sqrt(1 - rho^2);

    [actualPoints, weights] = gaussnorm(n, mu, baseSigma^2);

    transition = zeros(n, n);
    for i = 1:n
        for j = 1:n
            ymean = actualPoints(j) - rho * actualPoints(i) - (1 - rho) * mu;
            yunc  = actualPoints(j) - mu;
            
            transProb  = normpdf(ymean, 0, sigma);
            uncondProb = normpdf(yunc, 0, baseSigma);
            
            transition(i, j) = weights(j) * (transProb / uncondProb);
        end
    end

    % Normalize rows
    probabilities = transition ./ sum(transition, 2);
end
