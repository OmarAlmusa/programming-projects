function Trainer_3d_matrix(X, Y, network, epochs, learning_rate, beta)
for e = 1:epochs
    error = 0;
    %if length(size(X)) == 3
    %   v = [];
    %   for i = 1:length(X)
    %       v(i) = reshape(X(:, :, i), 25, 1);
    %   end
    %   X = v;
    %end
    for i = 1:length(size(X, 1))
        tensor = reshape(X(:, :, i), 25, 1);
        target = Y(:, i);
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
        end
        error = error + CrossEntropy(target, tensor);
        
        grad = CrossEntropy_prime(target, tensor);
        
        flipped_nn = flip(network);
        for layer = 1:length(flipped_nn)
            grad = flipped_nn{layer}.backward(grad, learning_rate, beta);
        end
    end
    if mod(e, 1000) == 0
        fprintf('\nepoch: %d / %d \t error: %f\n', e, epochs, error);
    end
end
end
