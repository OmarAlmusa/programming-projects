classdef Dropout < handle
    
    properties
        in
        ratio
        out
    end
    
    methods
        function obj = Dropout(ratio)
            obj.ratio = ratio;
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            [m, n] = size(obj.in);
            drop = zeros(m, n);
            
            num = round(m*n*(1-obj.ratio));
            idx = randperm(m*n, num);
            drop(idx) = 1 / (1-obj.ratio);
            obj.out = obj.in.*drop;
            output = obj.out;
            
        end
        
        function output = backward(obj, grad, learning_rate)
            output = grad;
        end
        
        function update(obj)
            %do nothing
        end
    end
end

