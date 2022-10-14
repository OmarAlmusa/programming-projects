cnn_network_gpu = {Convolution_gpu([1, 28, 28], [3, 3], 5, 'HE');
                   BatchNorm(prod([1, 28, 28]), 'HE')
                   ReLU();
                   MaxPooling([5, 26, 26], [2, 2], 2);
                   Reshape([5, 8, 8], [5*8*8, 1]);
                   Dense_gpu(5*8*8, 64, 'HE');
                   ReLU();
                   Dense_gpu(64, 10, 'NX');
                   Softmax()};