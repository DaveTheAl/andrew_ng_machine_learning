function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Cost without regularization
where_R = X*Theta' - Y;
J = 0.5 * sum(sum( ( where_R(R) ).^2 ));

%Gradient withouth regularization
inner = X*Theta' - Y;
for t=1:length(size(Theta, 1))
    Theta_grad(t) = sum(inner*Theta(t,:) .* R(t,:));
end

for t=1:length(size(X, 1))
    X_grad(t) = sum(inner*X(t,:) .* R(t,:));
end


% inner = X*Theta' - Y;
% disp('X_grad stuff:');
% disp(size(X_grad));
% disp(size(inner));
% disp(size(X));
% 
% disp('Theta_grad stuff:');
% disp(size(Theta_grad));
% disp(size(inner));
% disp(size(Theta));

% a = inner'*X;
% disp('a');
% disp(size(a));
% b = inner*Theta;
% disp('b');
% disp(size(b));
% 
% c = sum(a);
% disp('c');
% disp(size(c));
% disp(size(X_grad));
% d = sum(b);
% disp('d');
% disp(size(d));
% disp(size(Theta_grad));
% 
% X_grad = sum( inner'*X .* R, 2 );
% Theta_grad = sum( inner*Theta .* R, 2 ); 


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
