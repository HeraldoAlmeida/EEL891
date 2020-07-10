#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados BOSTON (problema de regressÃ£o)
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Boston em um dataframe do pandas
#------------------------------------------------------------------------------
    
import pandas as pd
dataframe = pd.read_excel('../../data/D02_Boston.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
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
#  Aplicar uma escala-padrão às variáveis de entrada do modelo
#------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

#------------------------------------------------------------------------------
#  Treinar um modelo de regressao linear
#------------------------------------------------------------------------------

print("----------------------------- ")
print(" LINEAR REGRESSION")
print("----------------------------- ")

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
from sklearn.metrics import mean_squared_error

mse_in  = mean_squared_error ( y_train , y_pred_train)
mse_out = mean_squared_error ( y_test  , y_pred_test)

rmse_in  = math.sqrt(mse_in)
rmse_out = math.sqrt(mse_out)

print ( 'RMSE   %10.4f  %10.4f' % ( rmse_in , rmse_out ) )

#------------------------------------------------------------------------------
#  Treinar um modelo de regressao KNN
#------------------------------------------------------------------------------

print("----------------------------- ")
print(" KNN")
print("----------------------------- ")

from sklearn.neighbors import KNeighborsRegressor

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

#------------------------------------------------------------------------------
#  Treinar um modelo de regressao polinomial
#------------------------------------------------------------------------------

print("----------------------------- ")
print(" POLINOMIAL REGRESSION")
print("----------------------------- ")

from sklearn.preprocessing import PolynomialFeatures

for grau in range(1,4):

    poly = PolynomialFeatures(degree=grau)
    poly.fit(X_train)
    
    Z_train = poly.transform(X_train)
    Z_test  = poly.transform(X_test)
    
    modelo = LinearRegression(
        fit_intercept = True,
        normalize     = False,
        copy_X        = True,
        n_jobs        = -1
        )
       
    modelo = modelo.fit(Z_train, y_train)
    
    y_pred_train = modelo.predict(Z_train)
    y_pred_test  = modelo.predict(Z_test)
    
    mse_in  = mean_squared_error ( y_train , y_pred_train)
    mse_out = mean_squared_error ( y_test  , y_pred_test)

    rmse_in  = math.sqrt(mse_in)
    rmse_out = math.sqrt(mse_out)
    
    print ( '%2d  %10.4f  %10.4f' % ( grau , rmse_in , rmse_out ) )

    