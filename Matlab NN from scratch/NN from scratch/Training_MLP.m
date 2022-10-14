Load_data

MLP_network

s = size(Images);

len = 8000;

epochs = 5;
learning_rate = 0.01;
beta = 0;
nn_network = Network;

for e=1:epochs
    error = 0;
    for data=1:len
        img = reshape(squeeze(Images(data, 1, :, :)), 28*28, 1);
        label = Labels_OHE(:, data);
        
        %forward prop: **DHW_format**
        x = img;
        for layer=1:length(nn_network)
            x = nn_network{layer}.forward(x);
        end
        
        error = error + CrossEntropy(label, x);
        
        grad = CrossEntropy_prime(label, x);
        
        flipped_nn = flip(nn_network);
        for layer=1:length(flipped_nn)
            grad = flipped_nn{layer}.backward(grad, learning_rate, beta);
        end
        
        if mod(data, 1000) == 0
            fprintf('reached_training: %d/%d\n', data, len);
        end
    end
    error = error / len;
    fprintf('\n%d/%d \t error: %.3f\n', e, epochs, error);
end