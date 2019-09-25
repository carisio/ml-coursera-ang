function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% As linhas de X são no formato [1 x1 x2]
% Theta é uma coluna com [theta0; theta1; theta2]
% Temp são todos os thetas exceto o theta 0 (para regularização)
% Versão vetorizada:
temp = theta; temp(1) = 0;

h_teta_x = sigmoid(X*theta);

J = 1/m * sum(-y.*log(h_teta_x) - (1-y).*log(1 - h_teta_x) ) ...
    + lambda/2/m * sum(temp.^2);

grad = 1/m * ( (h_teta_x - y)' * X) ...
      + (lambda/m * temp)';

% Versão com o gradiente não vetorizado
%nThetas = length(theta);
%
%h_teta_x = sigmoid(X*theta);
%J = 1/m * sum(-y.*log(h_teta_x) - (1-y).*log(1 - h_teta_x) ) ...
%    + lambda/2/m * sum(theta(2:nThetas).^2);
%
%for i=1:nThetas
%  grad(i) = 1/m * sum( (h_teta_x - y).*X(:,i) ) + (i ~= 1)*lambda/m * theta(i);
%end

% =============================================================

end
