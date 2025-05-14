function [x, w] = qnwnorm1(n)
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
