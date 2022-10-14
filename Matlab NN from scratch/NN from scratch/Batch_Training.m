Load_data

MLP_with_BN

nn_network = BN_Network;

len = 6000;

batch_size = 50;
learning_rate = 1;
epochs = 10;

for e=1:epochs
    i = 0;
    
    while((i+1)*batch_size <= len)
        error = 0;
        grad = 0;
        %zero_grad
        for layer=1:length(nn_network)
            nn_network{layer}.zero_grad();
        end
        %done
        
        for bs=i*batch_size+1:(i+1)*batch_size
            img = reshape(squeeze(Images(bs, 1, :, :)), 28*28, 1);
            label = Labels_OHE(:, bs);
            
            %forwardprop:
            x = img;
            for layer=1:length(nn_network)
                x = nn_network{layer}.forward(x);
            end
            %done
            
            error = error + CrossEntropy(label, x);
            
            
            grad = CrossEntropy_prime(label, x);
            
            %backwardprop:
            flipped_nn = flip(nn_network);
            for layer=1:length(flipped_nn)
                grad = flipped_nn{layer}.backward(grad, learning_rate);
            end
            %done
            
        end
        
        error = error ./ batch_size;
        
        %Update_parameters:
        for layer=1:length(nn_network)
            nn_network{layer}.update(batch_size);
        end
        %done
        
        if(mod((i+1)*batch_size, 1000) == 0)
            fprintf('\nepoch:%d/%d \t batch:%d/%d \t error:%.4f\n\n', e, epochs, (i+1)*batch_size, len, error);
        end
        
        i = i+1;
    end
        
    
end
