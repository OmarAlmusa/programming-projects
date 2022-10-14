classdef Convolution < handle

    properties
        in
        out
        input_shape
        input_depth
        depth
        output_shape
        kernel_shape
        weights
        bias
        dw
        dx
        db
    end
    
    methods
        function obj = Convolution(input_shape, kernel_size, depth, initialization_type)
            type = initialization_type;
            
            in_s = prod(input_shape);
            n_size = prod(kernel_size);
            
            input_depth = input_shape(1);
            input_height = input_shape(2);
            input_width = input_shape(3);
            obj.depth = depth;
            obj.input_shape = input_shape;
            obj.input_depth = input_depth;
            obj.output_shape = [depth, input_height - kernel_size(1) + 1, input_width - kernel_size(2) + 1];
            obj.kernel_shape = [depth, input_depth, kernel_size(1), kernel_size(2)];
            
            obj.dw = zeros(obj.kernel_shape);
            obj.db = zeros(obj.output_shape);
                
            if type == 'NX'
                obj.weights = Normalized_Xavier(in_s, n_size, obj.kernel_shape);
                obj.bias = Normalized_Xavier(in_s, n_size, obj.output_shape);
                
            elseif type == 'HE'
                obj.weights = He_Weight(in_s, obj.kernel_shape);
                obj.bias = He_Weight(in_s, obj.output_shape);
                
            else
                obj.weights = Xavier(in_s, obj.kernel_shape);
                obj.bias = Xavier(in_s, obj.output_shape);
            end
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.out = obj.bias;
            for i=1:obj.depth
                for j=1:obj.input_depth
                    oper = squeeze(obj.out(i, :, :)) + conv2(squeeze(obj.in(j, :, :)), rot180(squeeze(obj.weights(i, j, :, :))), 'valid');
                    obj.out(i, :, :) = reshape(oper, 1, obj.output_shape(2), obj.output_shape(3));
                end
            end
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            obj.dx = zeros(obj.input_shape);
            
            for i=1:obj.depth
                for j=1:obj.input_depth
                    oper2 = conv2(squeeze(obj.in(j, :, :)), rot180(squeeze(grad(i, :, :))), 'valid');
                    obj.dw(i, j, :, :) = reshape(oper2, 1, 1, obj.kernel_shape(3), obj.kernel_shape(4));
                    oper3 = squeeze(obj.dx(j, :, :)) + conv2(squeeze(grad(i, :, :)), squeeze(obj.weights(i, j, :, :)), 'full');
                    obj.dx(j, :, :) = reshape(oper3, 1, obj.input_shape(2), obj.input_shape(3));
                end
            end
            
            obj.db = obj.db + learning_rate.*grad;
            obj.dw = obj.dw + learning_rate.*obj.dw;
            obj.dx = learning_rate.*obj.dx;
            
            
            
            output = obj.dx;
        end
        
        function update(obj, batch_size)
            obj.weights = obj.weights - (obj.dw ./ batch_size);
            obj.bias = obj.bias - (obj.db ./ batch_size);
        end
        
        function zero_grad(obj)
            obj.dw = zeros(obj.kernel_shape);
            obj.db = zeros(obj.output_shape);
        end
    end
end

