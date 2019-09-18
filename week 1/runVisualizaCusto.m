clear all
close all
clc

% Teste para visualizar fun��o custo de uma regress�o linear simples, em 1D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dados para regress�o
m = 100;
x = rand(m,1);
y = 3*x + rand(m,1);

figure(1);
plot(x,y,'o');
title('Dados para regress�o. Intercepto = 0, regress�o do tipo h = teta*x');
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
title('Fun��o custo');
xlabel('Teta1');
ylabel('Teta0');
zlabel('Custo');
figure(3);
contour(teta1,teta0,J,50);
title('Fun��o custo');
xlabel('Teta1');
ylabel('Teta0');
zlabel('Custo');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imprime a informa��o do menor teta encontrado varrendo os dados
% e a informa��o do teta correto pensando em uma regress�o linear 
tetaCorreto = polyfit(x,y,1);
disp(sprintf('Regress�o correta: %d x + %d', tetaCorreto(1), tetaCorreto(2)))
hold on
plot(tetaCorreto(1), tetaCorreto(2), 'xr', 'LineWidth', 3)