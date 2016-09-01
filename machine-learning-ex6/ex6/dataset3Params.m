function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

choices_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
choices_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
max = 100000;
for i_c = 1:length(choices_C)
    for i_s = 1:length(choices_sigma)
        %first train on X and y
        model = svmTrain(X, y, choices_C(i_c), @(x1, x2) gaussianKernel(x1, x2, choices_sigma(i_s))); 
        
        %choose C and sigma based on X_val and y_val
        predictions = svmPredict(model, Xval);
        tmp = mean(double(predictions ~= yval));
        
        if max > tmp
            C = choices_C(i_c);
            sigma = choices_sigma(i_s);
            max = tmp;
        end
    end
end


% =========================================================================

end
