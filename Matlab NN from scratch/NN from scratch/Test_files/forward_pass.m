function output = forward_pass(network, X)
    result = [];
    
    for i = 1:length(X)
        tensor = transpose(X(i, :));
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
        end
        result = [result; tensor];
    end
    output = transpose(result);
end

