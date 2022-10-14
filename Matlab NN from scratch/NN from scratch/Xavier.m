function output = Xavier(n, shape)
    %n = number of nodes in the previous layer
    lower = -(1.0 / sqrt(n));
    upper = (1.0 / sqrt(n));
    
    numbers = rand(shape);
    
    output = lower + numbers .* (upper - lower);
end

