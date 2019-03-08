function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Cost (calculating J)
%for i = 1 : m
%    J = J + (- y(i) * log(sigmoid(X(i))) - (1 - y(i))* log(1 - sigmoid(X(i))));
%end;

%J = J / m; 
h = sigmoid(X*theta);
% why (X*theta), isn't it theta'*X(?, not fit in dimensions)
J = (-y'*log(h) - (1-y)' * log(1-h)) / m;
%J = (-y'*log(h)-(1-y)'*log(1-h))/m;

% Grad (calculating the partial derivatives)

% gradTemp = zeros(size(theta));
% 
% 
% for j = 1 : size(theta)
%     temp  = 0;
%     for i = 1 : m
%         temp = temp + (sigmoid(X(i)) - y(i)) * X(i); %X(j)? right?
%     end
%     gradTemp(j) = temp / m;
% end
% 
% grad = gradTemp;
%     


grad = X'*(h-y)/m;


% =============================================================

end
