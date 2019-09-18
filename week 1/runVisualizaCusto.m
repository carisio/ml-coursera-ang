clear all
close all
clc

% Teste para visualizar função custo de uma regressão linear simples, em 1D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dados para regressão
m = 100;
x = rand(m,1);
y = 3*x + rand(m,1);

figure(1);
plot(x,y,'o');
title('Dados para regressão. Intercepto = 0, regressão do tipo h = teta*x');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custo para diferentes tetas
teta0 = [-5:0.1:5];
teta1 = [-10:0.1:10];
nTetas0 = length(teta0);
nTetas1 = length(teta1);
J = zeros(nTetas0, nTetas1);

for i=1:nTetas0
  for j=1:nTetas1
    J(i, j) = (1/2/m) * sum((teta0(i) + teta1(j)*x - y).^2);
  end
end
figure(2);
mesh(teta1, teta0, J);
title('Função custo');
xlabel('Teta1');
ylabel('Teta0');
zlabel('Custo');
figure(3);
contour(teta1,teta0,J,50);
title('Função custo');
xlabel('Teta1');
ylabel('Teta0');
zlabel('Custo');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imprime a informação do menor teta encontrado varrendo os dados
% e a informação do teta correto pensando em uma regressão linear 
tetaCorreto = polyfit(x,y,1);
disp(sprintf('Regressão correta: %d x + %d', tetaCorreto(1), tetaCorreto(2)))
hold on
plot(tetaCorreto(1), tetaCorreto(2), 'xr', 'LineWidth', 3)