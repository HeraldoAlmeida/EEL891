#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados BOSTON (problema de regressão)
#==============================================================================

from scipy.stats import pearsonr

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Boston em um dataframe do pandas
#------------------------------------------------------------------------------
    
import pandas as pd
dataframe = pd.read_excel('../../data/D02_Boston.xlsx')

#------------------------------------------------------------------------------
#  Verificar os nomes das colunas disponíveis
#------------------------------------------------------------------------------

columnNames = dataframe.columns

print("Nomes das colunas:\n")
print(columnNames)

#------------------------------------------------------------------------------
#  Imprimir o grafico de dispersao do alvo em relacao ao atributo 'LSTAT'
#------------------------------------------------------------------------------

var = 'CRIM'

dataframe.plot.scatter(x=var,y='target')
        
print ( "pearson coef = " , pearsonr(dataframe[var],dataframe['target']))

#------------------------------------------------------------------------------
#  Imprimir os gráficos de dispersão do alvo em relação a acda atributo
#------------------------------------------------------------------------------

for col in columnNames:
    dataframe.plot.scatter(x=col,y='target')

for col in columnNames:
    print ( "pearson(%7s)" % col , ") = %6.3f , %6.3e" % pearsonr(dataframe[col],dataframe['target']))

    
print ( "pearson coef = " , pearsonr(dataframe['RM'],dataframe['LSTAT']))
    
    
#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:, 1:14].values

X = dataframe.loc[ : , [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD', 
    'TAX', 
    'PTRATIO', 
    'B', 
    'LSTAT'
    ] ].values

#X = dataframe.loc[ : , ['CHAS','DIS','B'] ].values
y = dataframe.iloc[:, 14].values

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 200,
        random_state = 20190411
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
#  Verificar os par�metros do regressor
#------------------------------------------------------------------------------

import numpy as np

print('\nParametros do regressor:\n', 
      np.append( regressor.intercept_  , regressor.coef_ ) )
   