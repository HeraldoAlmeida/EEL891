#==============================================================================
#  Regressao Linear Multivariavel
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_csv('../data/D05_Startups.csv')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#------------------------------------------------------------------------------
# Codificar o atributo categorico (state)
#------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# primeiramente codificar em valores inteiros

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


# em seguida criar uma variavel binaria para cada valor

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Eliminar a variável binária redundante

X = X[:, 1:]

#------------------------------------------------------------------------------
# Dividir o conjunto de dados em treinamento e teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size = 0.4 #,
        #random_state = 2018
)

#------------------------------------------------------------------------------
# Ajustar escala dos atributos
# (desnecessario para regressao linear - o regressor ja faz isso internamente)
#------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler,MinMaxScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

#------------------------------------------------------------------------------
# Instanciar o regressor linear
#------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

LinearRegression()

#------------------------------------------------------------------------------
# Treinar o regressor linear usando o conjunto de treinamento
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
#print('MSE  = %.3f' %           mean_squared_error(y_train, y_pred_train) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )

print('\nDesempenho no conjunto de teste:')
#print('MSE  = %.3f' %           mean_squared_error(y_test , y_pred_test) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_test , y_pred_test)))
print('R2   = %.3f' %                     r2_score(y_test , y_pred_test) )

#------------------------------------------------------------------------------
#  Verificar os parâmetros do regressor
#------------------------------------------------------------------------------

import numpy as np

print('\nParametros do regressor:\n', 
      np.append( regressor.intercept_ , regressor.coef_  ) )


