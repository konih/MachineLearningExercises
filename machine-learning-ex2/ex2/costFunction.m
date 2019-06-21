function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

htheta = arrayfun(@sigmoid,X*theta);
J = sum(-y.*log(htheta)+(y-1).*log(1-htheta))/m;

lengthTheta = length(theta);
grad = zeros(lengthTheta,1);
for j = 1:lengthTheta;
  temp = 0;
  for i = 1:m
    temp += (sigmoid(theta'*X(i,:)')-y(i))*X(i,j);
  endfor
  temp/=m;
  grad(j)=temp;
endfor
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% =============================================================
end
