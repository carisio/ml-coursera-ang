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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K=num_labels;

% Faz o loop em todas as amostras do treinamento
for i=1:m
  % PROCESSAMENTO DA REDE (FORWARD PROPAGATION):
  %
  % CAMADA 1:
  % A saída da camada 1 é a própria entrada dos dados
  a_1 = X(i,:);
  % Aqui a_1 = um vetor linha com (features) colunas. Vamos fazer a
  % transposta para virar um vetor coluna com (features) linhas:
  a_1 = a_1';
  % Nesse ponto, não há o bias. Temos que colocar o bias para processar a 
  % próxima camada:
  a_1 = [1; a_1];
  
  % CAMADA 2:
  % A saída da camada 2 é a sigmóide da entrada da camada 1 multiplicada 
  % por cada peso correspondente.
  % Os pesos estão na variável Theta1, que organiza os dados da seguinte forma:
  % LINHA: cada linha corresponde a um nó da camada 2
  % COLUNA: cada coluna corresponde a um nó da camada 1
  % Ou seja, Theta1(j, i) liga o nó i da camada 1 ao nó j da camada 2
  % Assim, ao fazermos a multiplicação Theta1*a_1 teremos o resultado correto.
  % Por exemplo:
  % A linha 1 de Theta1 contém os pesos que ligam todos os nós a_1 ao nó 1 da
  % camada 2. A multiplicação Theta1*a_1 irá gerar um vetor coluna z_2. 
  % O elemento 1 desse vetor será theta1(1, 1)*a_1(1) + ... 
  %                                                   + theta1(401, 1)*a_1(401)
  z_2 = Theta1*a_1;
  % A saída da camada é a sigmóide de z_2:
  a_2 = sigmoid(z_2);
  % Aqui a saída ainda não contém o bias. Devemos inserí-lo para servir de
  % entrada para a próxima camada
  a_2 = [1; a_2];
  
  % CAMADA 3:
  % O processamento da camada 3 é similar ao da camada 2. A principal diferença
  % é que, como já é a saída, não inserimos o bias:
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3);
  
  % CÁLCULO DO CUSTO
  % Feito o processamento das camadas, calculamos o custo.
  %
  % Primeiro, definimos uma variável para guardar y_i. É um vetor coluna de
  % tamanho K. O vetor é totalmente 0, exceto no nó correto, que deve ser 1.
  y_i = [1:K]' == y(i);
  % A conta correta é fazendo um loop da seguinte forma
  %for k=1:K
  %  J = J + 1/m*(-y_i(k) * log(a_3(k)) +...
  %          -(1-y_i(k))*log(1 - a_3(k)));
  %end
  % Versão sem loop
  J = J - (1/m)*( y_i' * log(a_3) + (1-y_i)' * log(1 - a_3) );
end

% Aplica regularização
sum_teta1_square = sum(sum(Theta1(:,2:end).^2));
sum_teta2_square = sum(sum(Theta2(:,2:end).^2));
J = J + (lambda/2/m)*(sum_teta1_square + sum_teta2_square);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
