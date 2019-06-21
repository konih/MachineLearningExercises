function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% You need to return the following variables correctly 

hthetaMinusY = X*theta - y;

J = sum((hthetaMinusY).^2)/(2*m) + (lambda/(2*m)) * sum(theta(2:end,:).^2);

grad = X'*hthetaMinusY/m + lambda/(2*m)*theta
% =========================================================================

grad = grad(:);

end
