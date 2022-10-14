classdef BatchNorm < handle

    properties
        in
        weights
        bias
        out
        z
        s
        dz
        dw
        db
        dx
        tensor_centered
    end
    
    methods
        function obj = BatchNorm(in_dim, initialization_type)
            type = initialization_type;
            
            obj.dw = zeros(1);
            obj.db = zeros(1);
                
            if type == 'NX'
                obj.weights = Normalized_Xavier(in_dim, 1, 1);
                obj.bias = Normalized_Xavier(in_dim, 1, 1);
            
            elseif type == 'HE'
                obj.weights = He_Weight(in_dim, 1);
                obj.bias = He_Weight(in_dim, 1);
                
            else
                obj.weights = Xavier(in_dim, 1);
                obj.bias = Xavier(in_dim, 1);
            end
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            epsilon = 1e-5;
            m = mean(obj.in);
            v = var(obj.in);
            obj.s = sqrt(v + epsilon);
            obj.tensor_centered = obj.in - m;
            obj.z = obj.tensor_centered ./ obj.s;
            %obj.out = obj.weights .* obj.z + obj.bias;
            output = obj.z;
        end
        
        function output = backward(obj, grad, learning_rate)
            %obj.dw = obj.dw + learning_rate.*(sum(grad .* obj.z));
            %obj.db = obj.db + learning_rate.*(sum(grad));
            
            %obj.dz = grad .* obj.weights;
            %obj.dx = (1/length(grad)) ./ (obj.s .* (length(grad) .* obj.dz - sum(obj.dz) - obj.z .* sum(obj.dz.*obj.z)));
            %output = obj.dx;
            output = grad;
        end
        
        function update(obj, batch_size)
            %obj.weights = obj.weights - (obj.dw ./ batch_size);
            %obj.bias = obj.bias - (obj.db ./ batch_size);
        end
        
        function zero_grad(obj)
            %obj.dw = zeros(1);
            %obj.db = zeros(1);
        end
    end
end

