function output = numerical_derivation(func, tensor, eta)
    if eta
        %do nothing
    else
       eta = 0.0000001; 
    end
    output = (func.forward(tensor + eta) - func.forward(tensor)) / eta;
end

