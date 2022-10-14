classdef Dense < handle
    
    properties
        in
        weights
        bias
        dw
        db
        dx
        out
        n_neurons
        in_dim
        %momentum_w
        %momentum_b
    end
    
    methods
        function obj = Dense(in_dim, n_neurons, initialization_type)
            type = initialization_type;
            obj.n_neurons = n_neurons;
            obj.in_dim = in_dim;
            obj.dw = zeros(obj.n_neurons, obj.in_dim);
            obj.db = zeros(obj.n_neurons, 1);
            
            if type == 'NX'
                obj.weights = Normalized_Xavier(in_dim, n_neurons, [n_neurons, in_dim]);
                obj.bias = Normalized_Xavier(in_dim, n_neurons, [n_neurons, 1]);
            
            elseif type == 'HE'
                obj.weights = He_Weight(in_dim, [n_neurons, in_dim]);
                obj.bias = He_Weight(in_dim, [n_neurons, 1]);
                
            else
                obj.weights = Xavier(in_dim, [n_neurons, in_dim]);
                obj.bias = Xavier(in_dim, [n_neurons, 1]);
            end
            
            %obj.momentum_w = 0;
            %obj.momentum_b = 0;
            
        end
        
        function output = forward(obj, tensor)
            obj.in = tensor;
            obj.out = obj.weights * obj.in + obj.bias;
            output = obj.out;
        end
        
        function output = backward(obj, grad, learning_rate)
            %if beta
            %    dw = grad * transpose(obj.in);
            %    obj.momentum_w = learning_rate*dw + beta*obj.momentum_w;
            %    obj.weights = obj.weights - obj.momentum_w;
            %    
            %    obj.momentum_b = learning_rate*grad + beta*obj.momentum_b;
            %    obj.bias = obj.bias - obj.momentum_b;
            %    output = transpose(obj.weights) * grad;
            %    
            %else
                obj.dw = obj.dw + learning_rate.*(grad * transpose(obj.in));
                obj.db = obj.db + learning_rate.*grad;
                obj.dx = transpose(obj.weights) * grad;
                output = obj.dx;
            %end
        end
        
        function update(obj, batch_size)
            obj.weights = obj.weights - (obj.dw ./ batch_size);
            obj.bias = obj.bias - (obj.db ./ batch_size);
        end
        
        function zero_grad(obj)
            obj.dw = zeros(obj.n_neurons, obj.in_dim);
            obj.db = zeros(obj.n_neurons, 1);
        end
    end
end

