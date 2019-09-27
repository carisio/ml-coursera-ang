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

% Quantidade de entradas. Aqui estou considerando tamb�m o bias
nEntradas = size(Theta1,2);
% Quantidade de entradas na camada oculta (intermedi�ria).
% Aqui estou considerando o bias
nNosOcultos = size(Theta2, 2);

% Theta1 tem tamanho 25 x 401, que corresponde a 25 n�s na segunda camada
%                            e 401 entradas (20x20 da imagem + 1 bias)
% Theta2 tem tamanho 10 x 26, que corresponde a 10 n�s de sa�da e 26 n�s
%                           da camada intermedi�ria (25 n�s de entrada + 1 bias)

% Para Theta1, a linha i cont�m todas as entradas (x0...x400) correspondentes
% ao n� intermedi�rio a_i.

% Para cada n� oculto (exceto o bias), computa a sa�da do n� oculto para
% determinado Theta1. Para isso, � necess�rio calcular a fun��o sigm�ide
% em cada n� a_i considerando o vetor Theta1 e a matriz X
%
% PASSO 1: Calculando a entrada para os n�s das camadas intermedi�rias:
% A matriz X tem m linhas e nEntradas-1 (ou seja, ela n�o considera o x0).
% Ent�o primeiro precisamos adicionar o x0 � matriz X:
X = [ones(m,1) X];
% A entrada de cada n� ser� z = Theta1*X. Em forma matricial, da forma como
% as matrizes est�o definidas, a conta � X * transposta(Theta1)
z = X*Theta1';

% PASSO 2: Calculada a entrada para cada n� da camada intermedi�ria, temos que
% calcular a sa�da desses n�s, que � a sigm�ide:
h = sigmoid(z);
% Nesse ponto, h tem a sa�da dos n�s a_1 at� a_25. Para avaliar a pr�xima camada
% devemos adicionar o bias (a_0):
h = [ones(m,1) h];

% PASSO 3: A vari�vel h ser� a entrada para cada n� da �ltima camada. O que
% temos que fazer aqui � repetir a ideia do passo 1. Podemos fazer em um 
% �nico passo:
y = sigmoid(h*Theta2');
% Aqui, y ser� uma matriz de tamanho m x 10 (S�o 10 classes que estamos 
% avaliando), com a sa�da de cada um dos n�s

% PASSO 4: Agora basta pegar o m�ximo de cada uma das classes e retornar a
% classe correta
[valor p] = max(y, [], 2);

% =========================================================================


end
