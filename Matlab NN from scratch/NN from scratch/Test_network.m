error = 0;
for data=8000:10000
        img = reshape(squeeze(Images(data, 1, :, :)), 28*28, 1);
        label = Labels_OHE(:, data);
        
        %forward prop: **DHW_format**
        x = img;
        for layer=1:length(nn_network)
            x = nn_network{layer}.forward(x);
        end
        
        error = error + CrossEntropy(label, x);
        
end
error = error / 2000;
fprintf('\nerror: %.3f\n', error);