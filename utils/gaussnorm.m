function [x, w] = gaussnorm(n, mu, s2)
    [x0, w0] = qnwnorm1(n);
    x = x0 * sqrt(2.0 * s2) + mu;
    w = w0 / sqrt(pi);
end

