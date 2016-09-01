function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
[imX, imY] = size(X);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

dist = zeros(size(X,1),1); %vector to measure smallest distances

%least squares distance
for i=1:imX             %for each centroid
    %initializing the first cluster assignment
    idx(i) = 1; 
    dist(i) = sum((X(i,:) - centroids(1,:)).^2);
    for k=2:K           %for each centroid
        %a = X(i, :);            %debugging
        %b = K(k, :);            %debugging
       	tmp = sum( (X(i, :) - centroids(k, :)).^2 );
        if tmp < dist(i)
            dist(i) = tmp;
            idx(i) = k;
        end
    end
end

% =============================================================

end

