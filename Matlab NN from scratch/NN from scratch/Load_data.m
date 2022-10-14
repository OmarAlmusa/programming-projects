Images = loadMNISTImages('MNIST/t10k-images-idx3-ubyte');
Images = transpose(Images);
Images = reshape(Images, [], 1, 28, 28);

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

Labels = loadMNISTLabels('MNIST/t10k-labels-idx1-ubyte');
Labels = Labels + 1;
Labels_OHE = OneHotEncoder(Labels, length(classes));