function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%IMPLEMENTING THE COST PART
%Calculating the vanilla cost
in_sigma = (X * theta - y).^2;
cost_vanilla = (1/(2 * m)) * sum(in_sigma, 1);

%Calculating the regularization term
% reg_term = theta(:, 2:end).^2;
reg = (lambda/(2 * m)) * sum(theta(2:end).^2);  %this works, but what's the problem with the one-upper line...?
J = cost_vanilla + reg;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%IMPLEMENTING THE COST PART
%Calculating Vanilla gradient
% disp(size(grad));
% disp(size(in_sigma));
% disp(size(X));
% disp(size(X));
% disp(size(in_sigma));
% disp('Old grad');
% disp(size(grad));
in_sigma = (X * theta - y);
grad = (1/m) * X' * in_sigma; % sum(repmat(in_sigma, 1, size(X, 2)) .* X); %from the slides, this is the vectorized version
% disp('New grad');
% disp(size(grad));
% disp(size(theta(2:end)));
% disp(size(grad(2:end)));
grad(2:end) = grad(2:end) + (lambda/m) .* theta(2:end); %adding the regularization term

% =========================================================================
grad = grad(:);

end
