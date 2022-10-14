function output = OneHotEncoder(labels, classes)
    output = zeros(classes, length(labels));
    for i=1:length(labels)
        output(labels(i), i) = 1;
    end
end

