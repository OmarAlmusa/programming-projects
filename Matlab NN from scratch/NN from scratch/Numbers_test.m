X = [0, 1, 1, 0, 0; 
     0, 0, 1, 0, 0;
     0, 0, 1, 0, 0;
     0, 0, 1, 0, 0;
     0, 1, 1, 1, 0];
     
X(:, :, 2) = [1, 1, 1, 1, 0;
              0, 0, 0, 0, 1;
              0, 1, 1, 1, 0;
              1, 0, 0, 0, 0;
              1, 1, 1, 1, 1];
     
X(:, :, 3) = [1, 1, 1, 1, 0;
              0, 0, 0, 0, 1;
              0, 1, 1, 1, 0;
              0, 0, 0, 0, 1;
              1, 1, 1, 1, 0];
     
X(:, :, 4) = [0, 0, 0, 1, 0;
              0, 0, 1, 1, 0;
              0, 1, 0, 1, 0;
              1, 1, 1, 1, 1;
              0, 0, 0, 1, 0];
     
X(:, :, 5) = [1, 1, 1, 1, 1;
              1, 0, 0, 0, 0;
              1, 1, 1, 1, 0;
              0, 0, 0, 0, 1;
              1, 1, 1, 1, 0];
Y = eye(5);