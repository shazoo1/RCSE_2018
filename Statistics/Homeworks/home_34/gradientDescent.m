function [ parameters, costHistory ] = gradientDescent( x, y, parameters, learningRate, repetition )
    %  Copied and adopted from https://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re

    % Getting the length of our dataset
    m = length(y);
    % Creating a matrix of zeros for storing our cost function history
    costHistory = zeros(repetition, 1);
    
    
    % Running gradient descent
    for i = 1:repetition        

        % Calculating the transpose of our hypothesis
        h = (x * parameters - y)';        

        % Updating the parameters
        parameters(1) = parameters(1) - learningRate * (1/m) * h * x(:, 1);

        parameters(2) = parameters(2) - learningRate * (1/m) * h * x(:, 2);        

        % Keeping track of the cost function
        costHistory(i) =  (x * parameters - y)' * (x * parameters - y) / (2 * length(y));

    end
   
end