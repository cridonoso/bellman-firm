function [stop_flag , V_guess, diff]= check_convergence(V_next, V_guess, iter, params)
% Check convergence of the value function iteration.
%
% This block compares the current value function guess `V_guess` with the
% updated value function `V_next`, using the maximum absolute difference
% across all grid points.
%
% If the maximum difference is below the specified tolerance (`params.tol`),
% the algorithm is considered to have converged:
%   - A message is displayed indicating the iteration at which convergence was achieved.
%   - The final value function (`V_guess`) and policy function (`policy_K`) are updated.
%   - The `stop` flag is set to true, signaling to terminate the main loop.
%
% If convergence is not yet reached, `stop` is set to false, allowing the
% algorithm to continue iterating.
    diff = max(abs(V_next(:) - V_guess(:)));
    if diff < params.tol
        disp(['Convergence achieved at iteration: ', num2str(iter), ', Difference: ', num2str(diff)]);
        V_guess = V_next; % save the best value
        stop_flag  = true;
    else
        stop_flag  = false;
        V_guess = V_next; % update guess for the next iteration
    end

    if iter == params.niter
        disp('Maximum number of iterations reached. No convergence achieved.');
    end

end

