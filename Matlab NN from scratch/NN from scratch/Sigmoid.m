classdef Sigmoid < handle
    
    properties
        sigmoid_func
        sigmoid_func_prime
        in
        out
        dx
        weights
        bias
    end
    
    methods
        function obj = Sigmoid()
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.sigmoid_func = 1 ./ (1+exp(-obj.in));
            obj.out = obj.sigmoid_func;
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            obj.sigmoid_func_prime = obj.sigmoid_func.*(1 - obj.sigmoid_func);
            obj.dx = grad.*obj.sigmoid_func_prime;
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

