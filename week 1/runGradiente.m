clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Faz o gradiente pra preencher uma regressão linear, plotando
% as curvas para diferentes valores de teta0 e teta1
teta0 = 1;
teta1 = 2;

x = 0:0.01:2;
y = teta1*x.*rand(size(x))  + teta0 + rand(size(x));

alfa = 0.01;

m = length(x);
teta0_ml = 10;
teta1_ml = 20;
iter_max = 10000;
n_plots = 20;

for t=1:iter_max
    Y_teta = teta1_ml*x + teta0_ml;
    teta0_ml = teta0_ml - alfa/m * sum((y - Y_teta).*(-1));
    teta1_ml = teta1_ml - alfa/m * sum((y - Y_teta).*(-x));
   
    if mod(t,iter_max/n_plots) == 0
       fprintf('%d, %d\n', teta0_ml, teta1_ml);
       figure(1)
       hold on
       plot(x,y,'.',x,(teta1_ml*x + teta0_ml),'-')
       pause
    end
end