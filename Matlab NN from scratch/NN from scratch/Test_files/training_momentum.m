X = [0, 0, 1;
     0, 1, 1;
     1, 0, 1;
     1, 1, 1];
 
Y = [0;
     1;
     1;
     0];
 
network = {Dense(3, 4);
           ReLU();
           Dense(4, 4);
           ReLU();
           Dense(4, 4);
           ReLU();
           Dense(4, 1);
           Sigmoid()};
 
epochs = 10000;
learning_rate = 0.3;
beta = 0.3;

%forward pass result before training:

error = 0;
for i = 1:length(X)
        tensor = transpose(X(i, :));
        target = Y(i);
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
            
        end
        fprintf('%f\n', tensor);
        error = error + mse(target, tensor);
end
fprintf('\nerror:%f\n\n', error);


%training:
for e = 1:epochs
    error = 0;
    for i = 1:length(X)
        tensor = transpose(X(i, :));
        target = Y(i);
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
        end
        error = error + mse(target, tensor);
        
        grad = mse_prime(target, tensor);
        
        flipped_nn = flip(network);
        for layer = 1:length(flipped_nn)
            grad = flipped_nn{layer}.backward(grad, learning_rate, beta);
        end
    end
    if mod(e, 1000) == 0
        fprintf('\nepoch: %d / %d \t error: %f\n', e, epochs, error);
    end
end


%forward pass result AFTER training:

error = 0;
for i = 1:length(X)
        tensor = transpose(X(i, :));
        target = Y(i);
        for layer = 1:length(network)
            tensor = network{layer}.forward(tensor);
            
        end
        fprintf('%f\n', tensor);
        error = error + mse(target, tensor);
end
fprintf('\nerror:%f\n\n', error);