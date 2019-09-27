function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Quantidade de entradas. Aqui estou considerando também o bias
nEntradas = size(Theta1,2);
% Quantidade de entradas na camada oculta (intermediária).
% Aqui estou considerando o bias
nNosOcultos = size(Theta2, 2);

% Theta1 tem tamanho 25 x 401, que corresponde a 25 nós na segunda camada
%                            e 401 entradas (20x20 da imagem + 1 bias)
% Theta2 tem tamanho 10 x 26, que corresponde a 10 nós de saída e 26 nós
%                           da camada intermediária (25 nós de entrada + 1 bias)

% Para Theta1, a linha i contém todas as entradas (x0...x400) correspondentes
% ao nó intermediário a_i.

% Para cada nó oculto (exceto o bias), computa a saída do nó oculto para
% determinado Theta1. Para isso, é necessário calcular a função sigmóide
% em cada nó a_i considerando o vetor Theta1 e a matriz X
%
% PASSO 1: Calculando a entrada para os nós das camadas intermediárias:
% A matriz X tem m linhas e nEntradas-1 (ou seja, ela não considera o x0).
% Então primeiro precisamos adicionar o x0 à matriz X:
X = [ones(m,1) X];
% A entrada de cada nó será z = Theta1*X. Em forma matricial, da forma como
% as matrizes estão definidas, a conta é X * transposta(Theta1)
z = X*Theta1';

% PASSO 2: Calculada a entrada para cada nó da camada intermediária, temos que
% calcular a saída desses nós, que é a sigmóide:
h = sigmoid(z);
% Nesse ponto, h tem a saída dos nós a_1 até a_25. Para avaliar a próxima camada
% devemos adicionar o bias (a_0):
h = [ones(m,1) h];

% PASSO 3: A variável h será a entrada para cada nó da última camada. O que
% temos que fazer aqui é repetir a ideia do passo 1. Podemos fazer em um 
% único passo:
y = sigmoid(h*Theta2');
% Aqui, y será uma matriz de tamanho m x 10 (São 10 classes que estamos 
% avaliando), com a saída de cada um dos nós

% PASSO 4: Agora basta pegar o máximo de cada uma das classes e retornar a
% classe correta
[valor p] = max(y, [], 2);

% =========================================================================


end
