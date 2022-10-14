function output = CrossEntropy_prime(target, nn_out)
    output = (((1 - target) ./ (1 - nn_out)) - (target ./ nn_out)) ./ length(target);
end

