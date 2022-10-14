network_1 = {Dense(25, 20);
    ReLU();
    Dropout(0.2);
    Dense(20, 20);
    ReLU();
    Dropout(0.2);
    Dense(20, 10);
    ReLU();
    Dropout(0.2);
    Dense(10, 5);
    Softmax()
    
};

network_2 = {Dense(25, 20);
    ReLU();

    Dense(20, 20);
    ReLU();
 
    Dense(20, 10);
    ReLU();

    Dense(10, 5);
    Softmax()
    
};