function output = CrossEntropy(target, nn_out)
    output = -mean(target.*log(nn_out) + (1 - target).*log(1 - nn_out));
end

