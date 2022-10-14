cnn_network_1 = {Convolution([1, 28, 28], [3, 3], 5, 'HE');
                 ReLU();
                 MaxPooling([5, 26, 26], [2, 2], 2);
                 Reshape([5, 8, 8], [5*8*8, 1]);
                 Dense(5*8*8, 64, 'HE');
                 ReLU();
                 Dense(64, 10, 'NX');
                 Softmax()};