function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
[imX, imY] = size(X);

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
%disp(size(X));
%disp(size(theta));
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
J = sum(inner, 1)/m;

inner = sigmoid(X * theta) - y;

tmp = sum(repmat(inner, 1, imY) .* X);
grad = (1/m) .* tmp';

% =============================================================

end