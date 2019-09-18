clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Princípios de classificação
%
% Cria uma base de dados para teste.
% A base criada aqui será de IDADE vs TEM_UMA_DETERMINADA_CARACTERISTICA
% x = idade
% y = tem uma determinada característica? y = 1 (SIM) ou y = 0 (NÃO)
% 
% A ideia é inventar alguns dados desse tipo
x = [0 4 6 9 12 17 20 25 36 55 60 63 71 77];
y = [0 0 0 0 0  1  0  1  1  1  0  1  1  1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plota os dados
figure(1);
plot(x(y==0),y(y==0),'ob', 'LineWidth', 2);
hold on
plot(x(y==1),y(y==1),'xr', 'LineWidth', 2);

xlabel('Idade');
ylabel('Tem determinada característica?');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Para classificação logistic, cria-se uma função
% hipótese diferente da regressão tradicional.
%
% Na regressão linear, pra 1 variável:
%
%           h = teta0 + teta11*x  (1)
%
% Na regressão logística, usa-se a função sigmóide para,
% entre outras coisas, garantir que o intervalo de saída
% fique entre 0 e 1. Assim, a função hipótese, para regressão
% linear em uma variável:
% 
%           h(z) = 1 / (1 + exp(-z))   (2)
%           z = teta0 + teta1*x        (3)
%
% Essa função tem um formato de S, sendo que vale 0.5 quando z = 0,
% tendendo a 0 para z -> -inf e tendendo a 1 para z -> +inf
%
% Dessa forma, a ideia é escolher um conjunto de teta que 
% faz um corte no limite de decisão para x. Para equações
% da forma (3), isso significa escolher um x de corte em que,
% para x > alfa, y = 1|0 e para x < alfa, y = 0|1
%
% A função custo tradicional (J) usada para regressão linear é
%
%           J = 1/2m * SOMATORIO( (h - y)^2 )    (4)
%
% Entretanto, essa função tem vários mínimos locais, ela é 
% não-convexa. Para vermos isso, podemos plotar essa função
% para diferentes valores de teta0 e teta1:
teta0 = -30:0.1:1;
teta1 = 0:0.01:1.5;

% Observe na figura 3 que há regiões de mínimos locais que, se o gradiente
% começasse ali, não chegaria no mínimo global
J = plotJMinQuad(teta0, teta1, x, y);

% Valores de teta que minimizam J
[minval, row] = min(min(J,[],2));
[minval, col] = min(min(J,[],1));
teta = [teta0(row) teta1(col)];
disp(sprintf('Tetas que minimizam: (teta0, teta1) = (%d, %d)', teta(1), teta(2)));

% Plota os dados de x/y e os valores calculados usando essa função de custo
figure;
hold on
plot(x, funSigmoid(teta(1) + teta(2)*x)>0.5, 'ok', 'LineWidth', 5);
plot(x(y==0),y(y==0),'ob', 'LineWidth', 2);
plot(x(y==1),y(y==1),'xr', 'LineWidth', 2);
title('Preto = estimado')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%