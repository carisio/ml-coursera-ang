function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%% PRIMEIRA VERS�O DE TREINAMENTO. RESOLVE USANDO O GRADIENTE
%% COM LOOP FOR
%%
%% O algoritmo abaixo resolve o exerc�cio. 
%% Com nMaxIter = 1.000, a precis�o chega a 93.22%.
%% Com nMaxIter = 5.000, a precis�o chega a 93.66%.
%% Com nMaxIter = 10.000, chega a 95.32%.
%%
%%nMaxIter = 1000;   % Quantidade m�xima de itera��es
%%alfa = 1;       % Taxa de aprendizado
%%
%%for i = 1:num_labels
%%  for iter=1:nMaxIter
%%    [J, grad] = lrCostFunction(all_theta(i,:)', X, y == i, lambda);
%%    all_theta(i,:) = all_theta(i,:) - alfa*grad;
%%    if (mod(iter,1000) == 0)
%%      disp(iter);
%%    end
%%  end
%%end

%% SEGUNDA VERS�O. TREINAMENTO DO GRADIENTE OTIMIZADO
%% USANDO A FUN��O fminunc
%% Com MaxIter = 10, a precis�o chega a 85.68%
%% Com MaxIter = 100, a precis�o chega a 94.98%
%% Com MaxIter = 500, a precis�o chega a 96.16%
%% Com MaxIter = 1.000, a precis�o chega a 96.16%
options = optimset('GradObj', 'on', 'MaxIter', 500);
for i = 1:num_labels
  [theta, cost] = ...
    fminunc(@(t)(lrCostFunction(t, X, y == i, lambda)), all_theta(i,:)', options);
  all_theta(i,:) = theta;
  disp(sprintf('Treinando classe %d',i)); 
end

% =========================================================================


end
