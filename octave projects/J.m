function [jVal, gradient] = J(theta);
  jVal = theta(1)*theta(2) + theta(2)*theta(3) + theta(1) + 3*theta(3);
  gradient = zeros(3,1);
  gradient(1) = theta(2) + 1;
  gradient(2) = theta(1) + theta(3);
  gradient(3) = theta(2) + 3;
  
  
end  