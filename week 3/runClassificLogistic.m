clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Roda uma classifica��o log�stica
%
% Cria uma base de dados para teste.
% A base criada aqui ser� de IDADE vs TEM_UMA_DETERMINADA_CARACTERISTICA
% x = idade
% y = tem uma determinada caracter�stica? y = 1 (SIM) ou y = 0 (N�O)
% 
% A ideia � inventar alguns dados desse tipo
x = [0 4 6 9 12 17 20 25 36 55 60 63 71 77]';
y = [0 0 0 0 0  1  0  1  1  1  0  1  1  1]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plota os dados
figure(1);
plot(x(y==0),y(y==0),'ob', 'LineWidth', 2);
hold on
plot(x(y==1),y(y==1),'xr', 'LineWidth', 2);

xlabel('Idade');
ylabel('Tem determinada caracter�stica?');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aplica o gradiente pra achar os pesos pra classifica��o
% log�stica
teta = [0 0];      % Valores inicias para teta
nMaxIter = 1000;   % Quantidade m�xima de itera��es
alfa = 0.01;       % Taxa de aprendizado
m = length(y);     % Quantidade de amostras da base
x = [ones(m, 1) x] % Adiciona uma coluna com 1 para multiplicar os teta0

for i = i:nMaxIter
  teta = teta - alfa/m * sum((funSigmoid(x*teta') - y).*x);
end

% Plota os dados de x/y e os valores calculados usando essa fun��o de custo
figure;
hold on
plot(x(:,2), funSigmoid(x*teta')>0.5, 'ok', 'LineWidth', 5);
plot(x(:,2)(y==0),y(y==0),'ob', 'LineWidth', 2);
plot(x(:,2)(y==1),y(y==1),'xr', 'LineWidth', 2);
title('Preto = estimado');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%