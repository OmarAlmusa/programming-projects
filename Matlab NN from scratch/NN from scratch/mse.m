function output = mse(target,nn_out)
    output = mean( (target - nn_out).^2);
end

