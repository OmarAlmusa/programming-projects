classdef Softmax < handle
    
    properties
        softmax
        in
        out
        weights
        bias
        dx
    end
    
    methods
        function obj = Softmax()
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.in = obj.in - max(obj.in);
            obj.in = obj.in;
            obj.softmax = exp(obj.in) ./ sum(exp(obj.in));
            obj.out = obj.softmax;
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            softmax_ = reshape(obj.softmax, 1, []);
            grad = reshape(grad, 1, []);
            
            d_softmax = ((softmax_ .* eye(length(softmax_))) - (transpose(softmax_) * softmax_));
            
            obj.dx = transpose(grad * d_softmax);
            output = obj.dx;
        end
        
        function update(obj, batch_size)
            %do nothing
        end
        
        function zero_grad(obj)
            %do nothing
        end
    end
end

