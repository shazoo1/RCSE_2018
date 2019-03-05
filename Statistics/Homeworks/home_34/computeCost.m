function J = computeCost(X, y, theta)

    % Taken, copied and adopted from https://ferdidolot.wordpress.com/2015/11/17/computing-gradient-descent-using-matlab/
   
    m = length(y); % number of training examples
 
    %calculate J function by using vectorized 
    J = sum((X *theta  - y).^2)/(2*m);
end