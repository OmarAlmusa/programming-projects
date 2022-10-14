function output = He_Weight(n, shape)
    %n = number of nodes in the previous layer
    dd = sqrt(2.0 / n);
    
    numbers = randn(shape);
    
    output = numbers .* dd;
end

