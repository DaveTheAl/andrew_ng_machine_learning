function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
[imX, imY] = size(X);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);

%The total equation is of the form
%sum( entrpoyPositive - entropyNegative )
f1  =-y .* log(h);
%disp(size(f1));
%disp(f1);

p = repmat(1, imX, 1) - y;
q = repmat(1, imX, 1) - h;

f2 = (p.*log(q));
%disp(size(f2));
%disp(f2);
inner = f1 - f2;
J = (sum(inner, 1)/m) + (lambda/m)* (1/2) * sum(theta(2:end).^2);

inner = sigmoid(X * theta) - y;

tmp = sum(repmat(inner, 1, imY) .* X);
%disp(tmp)
%disp(size(tmp(2:end)));
%disp(size(theta(2:end)));
tmp(2:end) = tmp(2:end) + lambda .* theta(2:end)';
grad = (1/m) .* tmp';





% =============================================================

end
