function output = forward_pass_3d(network, X)
    result = [];
    
    for i = 1:length(size(X, 1))
        tensor = reshape(X(:, :, i), 25, 1);
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
        end
        result = [result; tensor];
    end
    output = transpose(result);
end


