Network = {Dense(28*28, 32, 'HE');
            ReLU();
            Dense(32, 16, 'HE');
            ReLU();
            Dense(16, 10, 'XV');
            Softmax()};