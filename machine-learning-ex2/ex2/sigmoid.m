function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% As for z is a scalar: 
% g = 1 / (1 + exp(-z));

% As for z is a scalar/ vector / matrix:
 g = 1 ./ (1 + exp(-z)); % './' means use division for single element in the matrix


% =============================================================

end
