classdef ReLU < handle
    
    properties
        relu_func
        relu_func_prime
        in
        out
        weights
        bias
        dx
    end
    
    methods
        function obj = ReLU()
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.relu_func = max(obj.in, 0);
            obj.out = obj.relu_func;
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            obj.relu_func_prime = (obj.in > 0);
            obj.dx = obj.relu_func_prime.*grad;
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

