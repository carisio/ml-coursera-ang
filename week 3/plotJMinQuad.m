% Plota função de custo tradicional (mínimos quadrados)
% dado um h para diferentes valores de teta0
%
% Parâmetros de entrada:
% - Conjunto de valores para teta0
% - Conjunto de valores para teta1
% - x dado
% - y esperado
function [J] = plotJMinQuad(teta0, teta1, x, y)
  m = length(y);
  nTetas0 = length(teta0);
  nTetas1 = length(teta1);
  J = zeros(nTetas0, nTetas1);

  for i=1:nTetas0
    for j=1:nTetas1
      z = teta0(i) + teta1(j)*x;
      h_x = funSigmoid(z);
      J(i, j) = (1/2/m) * sum( (h_x - y).^2 );
    end
  end
  
  figure;
  mesh(teta1, teta0, J);
  title('Função custo');
  xlabel('Teta1');
  ylabel('Teta0');
  zlabel('Custo');
  
  figure;
  contour(teta1,teta0,J,50);
  title('Função custo');
  xlabel('Teta1');
  ylabel('Teta0');
  zlabel('Custo');
end
