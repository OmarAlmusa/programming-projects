function output = mse_prime(target, nn_output)
    output = 2 * (nn_output - target)./length(target);
end

