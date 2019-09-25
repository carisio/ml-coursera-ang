function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

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

% As linhas de X são no formato [1 x1 x2]
% Theta é uma coluna com [theta0; theta1; theta2]

h_teta_x = sigmoid(X*theta);
J = 1/m * sum(-y.*log(h_teta_x) - (1-y).*log(1 - h_teta_x) );

%nThetas = length(theta);

%for i=1:nThetas
%  grad(i) = 1/m * sum( (h_teta_x - y).*X(:,i) );
%end

% O código comentado acima pode ser vetorizado para:
% temp = (h_teta_x - y);
% grad = 1/m * (temp' * X);
% Ou, em uma linha:
grad = 1/m * ( (h_teta_x - y)' * X);

% =============================================================

end
