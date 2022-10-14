function output = Normalized_Xavier(n, m, shape)
    %n = number of nodes in the previous layer
    %m = number of nodes in the current layer
    lower = -(sqrt(6.0) / sqrt(n+m));
    upper = (sqrt(6.0) / sqrt(n+m));
    
    numbers = rand(shape);
    
    output = lower + numbers .* (upper - lower);
end

