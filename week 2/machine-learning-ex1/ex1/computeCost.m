function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 1/2/m * sum ((X*theta - y).^2);

% Visualização de uma hipótese com sigmoide (para logistic)
% m = length(y);
% J = 0;
% for i = 1:m
%   input = X(i,:) * theta;
%   h = 1/(1+exp(-input));
%   J = J + 0.5*(h-y(i))^2;
% end

% =========================================================================

end
