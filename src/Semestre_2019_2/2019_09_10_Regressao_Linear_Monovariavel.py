#==============================================================================
#  Regressao Linear Simples
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataset = pd.read_csv('../../data/D04_Salario_vs_AnosExperiencia.csv')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 0.5#,
        #random_state = 20190411
)

#------------------------------------------------------------------------------
#  Instanciar um regressor da classe LinearRegression
#------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(n_jobs=4,fit_intercept=True)

#------------------------------------------------------------------------------
#  Treinar o regressor usando o conjunto de treinamento
#------------------------------------------------------------------------------

regressor.fit(X_train, y_train)

#------------------------------------------------------------------------------
#  Obter respostas do modelo para os conjuntos de treinamento e de teste
#------------------------------------------------------------------------------

y_pred_train = regressor.predict(X_train)
y_pred_test  = regressor.predict(X_test )

#------------------------------------------------------------------------------
#  Verificar desempenho do regressor
#     - nos conjunto de treinamento ("in-sample")
#     - nos conjunto de teste   ("out-of-sample")
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho no conjunto de treinamento:')
print('MSE  = %.3f' %           mean_squared_error(y_train, y_pred_train) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )

print('\nDesempenho no conjunto de teste:')
print('MSE  = %.3f' %           mean_squared_error(y_test , y_pred_test) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )

#------------------------------------------------------------------------------
#  Verificar os parâmetros do regressor
#------------------------------------------------------------------------------

import numpy as np

print('\nParametros do regressor:\n', 
      np.append( regressor.intercept_  , regressor.coef_ ) )

#------------------------------------------------------------------------------
#  Visualizar o resultado do regressor
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test , y_test , color = 'green')
plt.scatter(X_test , y_pred_test , color = 'blue')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salario vs Anos de Experiencia')
plt.xlabel('Anos de Experiencia')
plt.ylabel('Salario')
plt.show()

