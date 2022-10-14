BN_Network = {Dense(28*28, 32, 'HE');
            BatchNorm(32, 'HE');
            ReLU();
            Dense(32, 16, 'HE');
            BatchNorm(16, 'HE');
            ReLU();
            Dense(16, 10, 'NX');
            Softmax()};