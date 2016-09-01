function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%naive forloop implementation
% for k = 1:K
%     tmp = zeros(1, 1);
%     for i = 1:size(idx,1)
%         if idx(i) == k
%             tmp = [tmp; idx(i)];
%         end
%     end
%     mean = 0;
%     for i = 1:length(tmp)
%         mean = mean + X(tmp(i),:);
%     end
%     centroids(k, :) = (1/length(tmp)) * mean;
% end

for k = 1:K
    mask = idx == k;
    count = sum(mask);
%     a = X(mask, :); %debugging
%     b = 1/count;    %debugging
%     c = sum(X(mask, :), 1);
%     disp(size(a));
%     disp(size(b));
%     disp(size(c));
%     centroids(k, :) = c * b;
    centroids(k, :) = (1/count) * sum(X(mask, :), 1);   %should sum over all rows (top to down merged)    
end

% =============================================================


end

