#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados BOSTON (problema de regress√£o)
#==============================================================================

from scipy.stats import pearsonr

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Boston em um dataframe do pandas
#------------------------------------------------------------------------------
    
import pandas as pd
dataframe = pd.read_excel('../../data/D02_Boston.xlsx')

#------------------------------------------------------------------------------
#  Verificar os nomes das colunas dispon√≠veis
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
#  Imprimir os gr√°ficos de dispers√£o do alvo em rela√ß√£o a acda atributo
#------------------------------------------------------------------------------

for col in columnNames:
    dataframe.plot.scatter(x=col,y='target')

for col in columnNames:
    print ( "pearson(%7s)" % col , ") = %6.3f , %6.3e" % pearsonr(dataframe[col],dataframe['target']))

    
print ( "pearson coef = " , pearsonr(dataframe['RM'],dataframe['LSTAT']))


#------------------------------------------------------------------------------
#  Criar os arrays numÈricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:,1:-1].to_numpy()
y = dataframe.iloc[:,-1].to_numpy()

#------------------------------------------------------------------------------
# Dividir os dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=100,
    random_state=20200705
    )

#------------------------------------------------------------------------------
#  Aplicar uma escala-padr„o ‡s vari·veis de entrada do modelo
#------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


#------------------------------------------------------------------------------
#  Treinar um modelo de regressao linear
#------------------------------------------------------------------------------

print(" ")
print("LINEAR REGRESSION")
print(" ")

from sklearn.linear_model import LinearRegression

modelo = LinearRegression(
    fit_intercept = True,
    normalize     = False,
    copy_X        = True,
    n_jobs        = -1
    )

modelo = modelo.fit(X_train, y_train)

y_pred_train = modelo.predict(X_train)
y_pred_test  = modelo.predict(X_test)

#------------------------------------------------------------------------------
#  Avaliar o desempenho do modelo
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

mse_in  = mean_squared_error ( y_train , y_pred_train)
mse_out = mean_squared_error ( y_test  , y_pred_test)

rmse_in  = math.sqrt(mse_in)
rmse_out = math.sqrt(mse_out)

r2_in  = r2_score ( y_train , y_pred_train)
r2_out = r2_score ( y_test  , y_pred_test)

print ( 'MSE    %10.4f  %10.4f' % (  mse_in ,  mse_out ) )
print ( 'RMSE   %10.4f  %10.4f' % ( rmse_in , rmse_out ) )
print ( 'R2     %10.4f  %10.4f' % (   r2_in ,   r2_out ) )

#------------------------------------------------------------------------------
#  Treinar um modelo de regressao KNN
#------------------------------------------------------------------------------

print(" ")
print("KNN")
print(" ")

from sklearn.neighbors import KNeighborsRegressor

modelo = KNeighborsRegressor(
    n_neighbors = 5,
    weights     = 'uniform',
    n_jobs      = -1
    )

modelo = modelo.fit(X_train, y_train)

y_pred_train = modelo.predict(X_train)
y_pred_test  = modelo.predict(X_test)

#------------------------------------------------------------------------------
#  Avaliar o desempenho do modelo
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

mse_in  = mean_squared_error ( y_train , y_pred_train)
mse_out = mean_squared_error ( y_test  , y_pred_test)

rmse_in  = math.sqrt(mse_in)
rmse_out = math.sqrt(mse_out)

r2_in  = r2_score ( y_train , y_pred_train)
r2_out = r2_score ( y_test  , y_pred_test)

print ( 'MSE    %10.4f  %10.4f' % (  mse_in ,  mse_out ) )
print ( 'RMSE   %10.4f  %10.4f' % ( rmse_in , rmse_out ) )
print ( 'R2     %10.4f  %10.4f' % (   r2_in ,   r2_out ) )

#------------------------------------------------------------------------------
#  Verificar a influencia do parametro k do KNN
#------------------------------------------------------------------------------

for k in range(1,21):

    modelo = KNeighborsRegressor(
        n_neighbors = k,
        weights     = 'uniform',
        n_jobs      = -1
        )
    
    modelo = modelo.fit(X_train, y_train)
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    mse_in  = mean_squared_error ( y_train , y_pred_train)
    mse_out = mean_squared_error ( y_test  , y_pred_test)

    rmse_in  = math.sqrt(mse_in)
    rmse_out = math.sqrt(mse_out)
    
    print ( '%2d  %10.4f  %10.4f' % ( k , rmse_in , rmse_out ) )
    