function [x, w] = qnwnorm1(n)
% QNWNORM1 Computes nodes and weights for Gauss-Hermite quadrature.
%
%   This function calculates the abscissas (nodes) x and weights w for
%   Gauss-Hermite quadrature with n points. This quadrature rule is used to
%   approximate integrals of the form:
%
%       integral_{-Inf}^{Inf} f(y) * exp(-y^2) dy approx sum_{i=1}^{n} w(i) * f(x(i))
%
%   The algorithm uses initial guesses for the roots of the nth Hermite
%   polynomial and refines them using Newton's method. The weights are
%   then calculated based on these nodes and the Hermite polynomials.
%   The nodes are symmetric around zero.
%
%   Args:
%       n (int): The number of quadrature points (nodes and weights) to compute.
%                Must be a positive integer.
%
%   Returns:
%       x (double column vector): The n abscissas (nodes) of the Gauss-Hermite
%                                 quadrature rule, sorted in ascending order. (n x 1).
%       w (double column vector): The corresponding n weights for the Gauss-Hermite
%                                 quadrature rule. (n x 1).
%
%   Details:
%       - The iteration for finding roots (Newton's method) continues until
%         the change in z is less than 1e-14 or a maximum number of iterations
%         (maxit = 100) is reached.
%       - The recurrence relation for Hermite polynomials (physicists' version)
%         H_{j+1}(z) = 2z H_j(z) - 2j H_{j-1}(z) is implicitly used, scaled.
%         The specific recurrence in the code:
%         p1 = z * sqrt(2/j) * p2 - sqrt((j-1)/j) * p3;
%         relates to orthonormal polynomials phi_j(z) = (2^j j! sqrt(pi))^(-1/2) H_j(z),
%         for which phi_j(z) = (z * sqrt(2/j)) * phi_{j-1}(z) - sqrt((j-1)/j) * phi_{j-2}(z) (approx).
%         p1 corresponds to phi_n(z), and pp to its derivative scaled by sqrt(n).
%       - pim4 = 1/pi^0.25 = pi^(-1/4) is H_0 / (2^0 * 0! * sqrt(pi))^(1/2) for orthonormal polynomials.
%         With H_0 = 1, this is (sqrt(pi))^(-1/2) = pi^(-1/4), suggesting p1, p2, p3 relate to
%         orthonormal Hermite polynomials.
%
%   Raises:
%       error: If the Newton's method does not converge within 'maxit' iterations.

    maxit = 100;
    pim4 = 1 / pi^0.25;
    m = floor((n + 1) / 2);
    x = zeros(n, 1);
    w = zeros(n, 1);

    for i = 1:m % iterate over n+1/2! 
        if i == 1
            z = sqrt(2 * n + 1) - 1.85575 * (2 * n + 1)^(-1/6);
        elseif i == 2
            z = z - 1.14 * (n^0.426) / z;
        elseif i == 3
            z = 1.86 * z + 0.86 * x(1);
        elseif i == 4
            z = 1.91 * z + 0.91 * x(2);
        else
            z = 2 * z + x(i - 2);
        end

        for its = 1:maxit
            p1 = pim4;
            p2 = 0;
            for j = 1:n
                p3 = p2;
                p2 = p1;
                p1 = z * sqrt(2 / j) * p2 - sqrt((j - 1) / j) * p3;
            end
            pp = sqrt(2 * n) * p2;
            z1 = z;
            z = z1 - p1 / pp;
            if abs(z - z1) < 1e-14
                break;
            end
        end

        if its == maxit
            error('qnwnorm1 did not converge');
        end

        x(n + 1 - i) = z;
        x(i) = -z;
        w(i) = 2 / (pp^2);
        w(n + 1 - i) = w(i);
    end
end
