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
  % A sa�da da camada 1 � a pr�pria entrada dos dados
  a_1 = X(i,:);
  % Aqui a_1 = um vetor linha com (features) colunas. Vamos fazer a
  % transposta para virar um vetor coluna com (features) linhas:
  a_1 = a_1';
  % Nesse ponto, n�o h� o bias. Temos que colocar o bias para processar a 
  % pr�xima camada:
  a_1 = [1; a_1];
  
  % CAMADA 2:
  % A sa�da da camada 2 � a sigm�ide da entrada da camada 1 multiplicada 
  % por cada peso correspondente.
  % Os pesos est�o na vari�vel Theta1, que organiza os dados da seguinte forma:
  % LINHA: cada linha corresponde a um n� da camada 2
  % COLUNA: cada coluna corresponde a um n� da camada 1
  % Ou seja, Theta1(j, i) liga o n� i da camada 1 ao n� j da camada 2
  % Assim, ao fazermos a multiplica��o Theta1*a_1 teremos o resultado correto.
  % Por exemplo:
  % A linha 1 de Theta1 cont�m os pesos que ligam todos os n�s a_1 ao n� 1 da
  % camada 2. A multiplica��o Theta1*a_1 ir� gerar um vetor coluna z_2. 
  % O elemento 1 desse vetor ser� theta1(1, 1)*a_1(1) + ... 
  %                                                   + theta1(401, 1)*a_1(401)
  z_2 = Theta1*a_1;
  % A sa�da da camada � a sigm�ide de z_2:
  a_2 = sigmoid(z_2);
  % Aqui a sa�da ainda n�o cont�m o bias. Devemos inser�-lo para servir de
  % entrada para a pr�xima camada
  a_2 = [1; a_2];
  
  % CAMADA 3:
  % O processamento da camada 3 � similar ao da camada 2. A principal diferen�a
  % � que, como j� � a sa�da, n�o inserimos o bias:
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3);
  
  % C�LCULO DO CUSTO
  % Feito o processamento das camadas, calculamos o custo.
  %
  % Primeiro, definimos uma vari�vel para guardar y_i. � um vetor coluna de
  % tamanho K. O vetor � totalmente 0, exceto no n� correto, que deve ser 1.
  y_i = [1:K]' == y(i);
  % A conta correta � fazendo um loop da seguinte forma
  %for k=1:K
  %  J = J + 1/m*(-y_i(k) * log(a_3(k)) +...
  %          -(1-y_i(k))*log(1 - a_3(k)));
  %end
  % Vers�o sem loop
  J = J - (1/m)*( y_i' * log(a_3) + (1-y_i)' * log(1 - a_3) );
  
  % C�LCULO DO BACKPROPAGATION
  %
  % C�LCULO DE delta_3
  % delta_3 � o erro gerado pelos n�s da camada 3 (sa�da de rede). Como essa �
  % a �ltima camada, o erro � simplesmente a diferen�a entre a sa�da da rede
  % e a sa�da desejada.
  delta_3 = a_3 - y_i;
  
  % C�LCULO DE delta_2
  % delta_2 � o erro gerado pelos n�s da camada 2 (camada intermedi�ria). Nesse
  % caso, o erro da camada 3 propaga de volta para a camada 2.
  % NOTA: A intui��o abaixo desconsidera a multiplica��o pela derivada.
  % Por exemplo, delta_2(1) = Theta2(1,2)*delta_3(1) + Theta2(2,2)*delta_3(2) +
  %                           + .... + Theta2(K,2)*delta_3(K)
  %
  % Observe que n�o considera o termo de bias!
  %
  % No nosso caso, Theta2 � organizado em K linhas e n_nos_intermediarios
  % colunas. Para fazer essa conta com opera��o de matrizes, precisamos usar
  % a transposta de Theta2. Assim, Theta2 ficar� organizado em 
  % n_nos_intermediarios linhas e K colunas. A opera��o 
  % transponse(Theta2)*delta_3 vai gerar o vetor delta_2 desejado
  % Depois disso, basta multiplicar cada termo pela derivada do gradiente
  delta_2 = (Theta2(:,2:end)' * delta_3).*sigmoidGradient(z_2);
    
  % A partir da�, computamos o DELTA para calcular o gradiente de cada Theta:
  % DELTA(camada L) = DELTA(camad L) + delta(camada L+1)*transposta(a(camada L))
  Theta1_grad = Theta1_grad + delta_2*(a_1');
  Theta2_grad = Theta2_grad + delta_3*(a_2');
end

% O c�lculo do gradiente � o delta dividido por m
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% Aplica regulariza��o
sum_teta1_square = sum(sum(Theta1(:,2:end).^2));
sum_teta2_square = sum(sum(Theta2(:,2:end).^2));
J = J + (lambda/2/m)*(sum_teta1_square + sum_teta2_square);

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
