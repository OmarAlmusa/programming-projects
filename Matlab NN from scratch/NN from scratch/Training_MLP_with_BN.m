Load_data

MLP_network

s = size(Images);

len = 8000;

batch_size = 1;
epochs = 10;
learning_rate = 0.01;
nn_network = Network;

for e=1:epochs
    error = 0;
    for data=1:len
        
        %zero_grad
        for layer=1:length(nn_network)
            nn_network{layer}.zero_grad();
        end
        %done
        
        img = reshape(squeeze(Images(data, 1, :, :)), 28*28, 1);
        label = Labels_OHE(:, data);
        
        %forward prop: **DHW_format**
        x = img;
        for layer=1:length(nn_network)
            x = nn_network{layer}.forward(x);
        end
        %done
        
        error = error + CrossEntropy(label, x);
        
        grad = CrossEntropy_prime(label, x);
        
        %backward_update
        flipped_nn = flip(nn_network);
        for layer=1:length(flipped_nn)
            grad = flipped_nn{layer}.backward(grad, learning_rate);
        end
        %done
        
        %Update_parameters:
        for layer=1:length(nn_network)
            nn_network{layer}.update(batch_size);
        end
        %done
        
        if mod(data, 1000) == 0
            fprintf('reached_training: %d/%d\n', data, len);
        end
    end
    error = error / len;
    fprintf('\n%d/%d \t error: %.3f\n', e, epochs, error);
end