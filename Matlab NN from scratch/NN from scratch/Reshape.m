classdef Reshape < handle
    
    properties
        in
        input_shape
        output_shape
        out
        weights
        bias
        dx
    end
    
    methods
        function obj = Reshape(input_shape, output_shape)
            obj.input_shape = input_shape;
            obj.output_shape = output_shape;
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.out = reshape(obj.in, obj.output_shape);
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            obj.dx = reshape(grad, obj.input_shape);
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

