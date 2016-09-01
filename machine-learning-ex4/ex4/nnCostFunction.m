function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% disp('Theta 1');
% disp(size(Theta1));
% disp('Theta 2');
% disp(size(Theta2));
% disp('X');
% disp(size(X));
% disp('y');
% disp(size(y));

z2 = [ones(size(X, 1), 1), X] * Theta1';    %Extending input by bias vector, and getting output
% disp('z2');
% disp(size(z2));
a2 = sigmoid(z2);
% disp('a2');
% disp(size(a2));

z3 = [ones(size(a2, 1), 1), a2] * Theta2'; %Extending input by bias vector, and getting output
% disp('z3');
% disp(size(z3));
a3 = sigmoid(z3);
% disp('a3');
% disp(size(a3));

first_sum = 0;
% disp('next');
for i=1:num_labels
    %choose y and a3 for which the result is true
    
    %convert y into a one-hot vector
    tmp_y = y == i;
    
    score = a3(:, i);
    firstterm = -tmp_y .* log(score);    % we want to have the input of y, where y is equal to the predicted class
                                                        % choose (as y),
                                                        % the value
                                                        % correpsonding to
                                                        % the label
                                                        
                                                        %y will be a value
                                                        %between 0 and 1,
                                                        %so we will
                                                        %calculate the
                                                        %cross entropy for
                                                        %each class

    s1 = repmat(1, size(tmp_y, 1), 1) - tmp_y;
    s2 = repmat(1, size(tmp_y, 1), 1) - score;
    secondterm = s1 .* log(s2);

    first_sum = first_sum + firstterm - secondterm;
end


regT1 = Theta1(:, 2:end).^2;
regT2 = Theta2(:, 2:end).^2;
% 
% disp(sum(regT1(:)));
% disp(sum(regT2(:)));

J = (sum(first_sum, 1)/m) + (lambda/m) * (1/2) * ( sum(regT1(:)) + sum(regT2(:)) );      %not adding the bias units...

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% disp('Sizes');
% disp(size(Theta1));
% disp(size(Theta2));
% waitforbuttonpress; 

for t = 1:m     %apply backpropagation for each training sample
    %Step one: Feedworward
    a1 = X(t, :);
    
    z2 = [ones(size(a1, 1), 1), a1] * Theta1';    %Extending input by bias vector, and getting output
    a2 = sigmoid(z2);

    z3 = [ones(size(a2, 1), 1), a2] * Theta2'; %Extending input by bias vector, and getting output
    a3 = sigmoid(z3);
    
    %Step two: First Delta Calculations
    %convert y into a one-hot vector!
%     disp('a3 then y');
%     disp(size(a3));
    
    yk = a3(:, y(t)) == y(t);
    yk = zeros(1,num_labels);
    yk(1, y(t)) = 1;
%     disp(size(yk));
    delta3 = a3 - yk;
    
    %Step three: Second Layer Delta Calculations
    delta2 = delta3 * Theta2 .* [1, sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    
    
    %Step Four: Accumulation
%     disp('first');
%     disp(size(Theta1_grad));
%     disp(size(delta2));
%     disp(size(a1));
%     
%     disp('sec');
%     disp(size(Theta2_grad));
%     disp(size(delta3));
%     disp(size(a2));
    
    Theta1_grad = Theta1_grad + delta2' * [1, a1];
    Theta2_grad = Theta2_grad + delta3' * [1, a2];    
     
end

%Removing the 'biased' gradient terms... (derivative would be zero...)

%Step Five: Unregularized Gradient
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;


% inner = sigmoid(X * theta) - y;
% 
% tmp = sum(repmat(inner, 1, imY) .* X);
% tmp(2:end) = tmp(2:end) + lambda .* theta(2:end)';

% grad = (1/m) .* tmp';


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%adding regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);




% h = sigmoid(X * theta);
% % disp(h);
% %The total equation is of the form
% %sum( entrpoyPositive - entropyNegative )
% f1  =-y .* log(h);
% 
% p = repmat(1, imX, 1) - y;
% q = repmat(1, imX, 1) - h;
% 
% f2 = (p.*log(q));
% inner = f1 - f2;
% J = (sum(inner, 1)/m) + (lambda/m)* (1/2) * sum(theta(2:end).^2);
% 
% inner = sigmoid(X * theta) - y;
% 
% tmp = sum(repmat(inner, 1, imY) .* X);
% tmp(2:end) = tmp(2:end) + lambda .* theta(2:end)';
% grad = (1/m) .* tmp';





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
