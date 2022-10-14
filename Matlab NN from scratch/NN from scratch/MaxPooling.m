classdef MaxPooling < handle
    
    properties
        in
        input_shape
        kernel_size
        strides
        output_shape
        out_h
        out_w
        out
        dx
        weights
        bias
        grad_shape
    end
    
    methods
        function obj = MaxPooling(input_shape, kernel_size, strides)
            obj.input_shape = input_shape;
            obj.kernel_size =  kernel_size;
            obj.strides = strides;
            obj.out_h = int32((obj.input_shape(2) - obj.kernel_size(1))/(obj.strides + 1));
            obj.out_w = int32((obj.input_shape(3) - obj.kernel_size(2))/(obj.strides + 1));
            obj.output_shape = [obj.input_shape(1), obj.out_h, obj.out_w];
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            s = obj.strides;
            obj.out = zeros(obj.output_shape);
            
            for depth=1:obj.input_shape(1)
                
                for i=1:obj.out_h
                    for j=1:obj.out_w
                        oper = max(max(squeeze(obj.in(depth, i*s:obj.kernel_size(1) + (i*s), j*s:obj.kernel_size(2) + (j*s)))));
                        obj.out(depth, i, j) = oper;
                    end
                end
                
            end
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            obj.grad_shape = size(grad);
            s = obj.strides;
            obj.dx = zeros(obj.input_shape);
            
            for depth=1:obj.grad_shape(1)
                
                for i=1:obj.grad_shape(2)
                    for j=1:obj.grad_shape(3)
                        tmp = obj.in(depth, i*s:obj.kernel_size(1) + (i*s), j*s:obj.kernel_size(2) + (j*s));
                        
                        mask = (tmp == max(max(squeeze(tmp))));
                        
                        obj.dx(depth, i*s:obj.kernel_size(1) + (i*s), j*s:obj.kernel_size(2) + (j*s)) = obj.dx(depth, i*s:obj.kernel_size(1) + (i*s), j*s:obj.kernel_size(2) + (j*s)) + grad(depth, i, j).*mask;
                    end
                end
                
            end
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

