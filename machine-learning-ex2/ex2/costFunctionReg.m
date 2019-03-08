function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

theta1 = theta;
theta1(1) = 0;

% Cost:
h = sigmoid(X*theta);
%J = (-y'*log(h)-(1-y)'*log(1-h))/m + lambda * sum(theta.^2)/ (2*m);
J = (-y'*log(h)-(1-y)'*log(1-h))/m + theta' * theta1 * lambda/ (2*m);

% Grad:
% when j = 0
% for j = 1:size(theta)
%     if j == 1
%         grad = X'*(h-y) /m;
%     else
%         grad = X'*(h-y) /m + theta(j)* lambda/m;
%     end
% end

grad = X'*(h-y) /m + theta1* lambda/m;





% =============================================================

end
