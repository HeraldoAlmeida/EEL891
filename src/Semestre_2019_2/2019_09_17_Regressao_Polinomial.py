#==============================================================================
#  Regressao Linear Polinomial
#==============================================================================

import numpy as np

DEGREE = 6

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataframe = pd.read_csv('../../data/D06_Salario_vs_Nivel.csv')

print('')
print('dataframe =')
print(dataframe)

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:, 1].values
y = dataframe.iloc[:, 2].values

print('')
print('X.shape =',X.shape)
print('y.shape =',y.shape)

#------------------------------------------------------------------------------
#  Como neste caso so existe 1 atributo, X foi criado como vetor e
#  precisa ser redimensionado como matriz de 1 coluna para que o
#  metodo "fit" do regressor possa saber que se tratam de 10 amostras
#  com 1 unico atributo (e nao 1 amostra com 10 atributos)
#------------------------------------------------------------------------------

X = X.reshape(-1,1)  # redimensionar de vetor-linha para vetor-coluna

print('')
print('X.shape =',X.shape)
print('y.shape =',y.shape)

#------------------------------------------------------------------------------
#  Visualizar as amostras em um diagrama de dispersao "Salario vs Nivel"
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.scatter(X, y, color = 'red')
plt.title('Amostras disponiveis para treinamento')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Treinar um regressor linear com o conjunto de dados inteiro
#------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()
regressor1.fit(X, y)

#------------------------------------------------------------------------------
#  Mostrar os parametros do regressor linear
#------------------------------------------------------------------------------

print('\nParametros do regressor linear:\n', 
      np.append( regressor1.intercept_ , regressor1.coef_  ) )

#------------------------------------------------------------------------------
#  Obter as respostas do regressor linear para o conjunto de treinamento
#------------------------------------------------------------------------------

y_pred_1 = regressor1.predict(X)

#------------------------------------------------------------------------------
#  Visualizar graficamente as respostas do regressor linear
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.scatter(X, y, color = 'red', alpha = 0.5)
plt.scatter(X, y_pred_1, color = 'blue', marker = 'x')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Criar um grid de 0.00 a 10.00 com passo de 0.01
#------------------------------------------------------------------------------

import numpy as np

X_grid = np.arange(1.00,10.01,0.01).reshape(-1,1)

print('')
print('X_grid =')
print(X_grid)

#------------------------------------------------------------------------------
#  Obter as respostas do regressor linear para cada ponto do grid
#------------------------------------------------------------------------------

y_grid = regressor1.predict(X_grid)

print('')
print('y_grid =')
print(y_grid)

#------------------------------------------------------------------------------
#  Visualizar graficamente o modelo de regressao linear
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.plot(X_grid, y_grid, color = 'blue')
plt.scatter(X, y, color = 'red')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Verificar desempenho do regressor LINEAR
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho do regressor linear:')
#print('MSE  = %.3f' %           mean_squared_error(y, y_grid) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_1)))
print('R2   = %.3f' %                     r2_score(y, y_pred_1) )

#------------------------------------------------------------------------------
#  Treinar um regressor polinomial com o conjunto de dados inteiro
#------------------------------------------------------------------------------

# gerar atributos (coeficientes) do polinomio de grau desejado

from sklearn.preprocessing import PolynomialFeatures

poly_feat = PolynomialFeatures(degree = DEGREE)
X_poly = poly_feat.fit_transform(X)

# treinar o regressor polinomial, ou seja,
# um regressor linear treinado com os atributos polinomiais
# derivados dos atributos originais das amostras

regressor2 = LinearRegression(fit_intercept=False)
regressor2.fit(X_poly, y)

#------------------------------------------------------------------------------
#  Mostrar os parametros do regressor polinomial
#------------------------------------------------------------------------------

print('\nParametros do regressor linear:\n', regressor2.coef_ )

#------------------------------------------------------------------------------
#  Obter as respostas do regressor polinomial para o conjunto de treinamento
#------------------------------------------------------------------------------

y_pred_2 = regressor2.predict(X_poly)

#------------------------------------------------------------------------------
#  Visualizar graficamente as respostas do regressor polinomial
#------------------------------------------------------------------------------

plt.scatter(X, y, color = 'red', alpha = 0.5)
plt.scatter(X, y_pred_2, color = 'blue', marker = 'x')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Obter as respostas do regressor polinomial para cada ponto do grid
#------------------------------------------------------------------------------

X_poly_grid = poly_feat.transform(X_grid)
y_grid = regressor2.predict(X_poly_grid)

#------------------------------------------------------------------------------
#  Visualizar graficamente o modelo de regressao polinomial
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.plot(X_grid, y_grid, color = 'blue')
plt.scatter(X, y, color = 'red')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Comparar o desempenho dos regressores LINEAR e POLINOMIAL
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho do regressor linear:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_1)))
print('R2   = %.3f' %                     r2_score(y, y_pred_1) )

print('\nDesempenho do regressor polinomial:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_2)))
print('R2   = %.3f' %                     r2_score(y, y_pred_2) )


#
## Polynomial Regression
#
## Importing the libraries
##import numpy as np
##import matplotlib.pyplot as plt
##import pandas as pd
##
##from sklearn.metrics import mean_squared_error, r2_score
##
### Importing the dataset
##dataset = pd.read_csv('../Salario_vs_Posicao.csv')
##X = dataset.iloc[:, 1:2].values
##y = dataset.iloc[:, 2].values
#
## Splitting the dataset into the Training set and Test set
##"""from sklearn.cross_validation import train_test_split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
##
### Feature Scaling
##"""from sklearn.preprocessing import StandardScaler
##sc_X = StandardScaler()
##X_train = sc_X.fit_transform(X_train)
##X_test = sc_X.transform(X_test)"""
#
## Fitting Linear Regression to the dataset
##from sklearn.linear_model import LinearRegression
##lin_reg = LinearRegression()
##lin_reg.fit(X, y)
##
### Fitting Polynomial Regression to the dataset
##from sklearn.preprocessing import PolynomialFeatures
##poly_reg = PolynomialFeatures(degree = 12)
##X_poly = poly_reg.fit_transform(X)
##poly_reg.fit(X_poly, y)
##lin_reg_2 = LinearRegression()
##lin_reg_2.fit(X_poly, y)
#
## Visualising the Linear Regression results
#plt.scatter(X, y, color = 'red')
#plt.plot(X, reg1.predict(X), color = 'blue')
#plt.title('Truth or Bluff (Linear Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the Polynomial Regression results
#plt.scatter(X, y, color = 'red')
#plt.plot(X, reg2.predict(poly_feat.fit_transform(X)), color = 'blue')
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the Polynomial Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, reg2.predict(poly_feat.fit_transform(X_grid)), color = 'blue')
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
#
## Predicting the Test set results
#
#y_pred_linear = reg1.predict(X)
#y_pred_poly = reg2.predict(poly_feat.fit_transform(X))
#
##------------------------------------------------------------------------------
##  Verificar desempenho do regressor
##     - nos conjunto de treinamento ("in-sample")
##     - nos conjunto de teste   ("out-of-sample")
##------------------------------------------------------------------------------
#
#import math
#from sklearn.metrics import mean_squared_error, r2_score
#
#print('\nDesempenho do regressor linear:')
#print('MSE  = %.3f' %           mean_squared_error(y, y_pred_linear) )
#print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_linear)))
#print('R2   = %.3f' %                     r2_score(y, y_pred_linear) )
#
#print('\nDesempenho no regressor polinomial:')
#print('MSE  = %.3f' %           mean_squared_error(y, y_pred_poly) )
#print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_poly)))
#print('R2   = %.3f' %                     r2_score(y, y_pred_poly) )
#
##------------------------------------------------------------------------------
##  Verificar os parâmetros do regressor
##------------------------------------------------------------------------------
#
#print('Parametros do Regressor Linear:     ', reg1.intercept_ , reg1.coef_)
#print('Parametros do Regressor Polinomial: ', reg2.intercept_ , reg2.coef_)
