import pandas as pd

#------------------------------------------------------
# LE ARQUIVOS CSV DE TREINO E DE TESTE
#------------------------------------------------------

df_treino = pd.read_csv('../../data/train.csv')
df_teste  = pd.read_csv('../../data/test.csv')


df_treino['preco'] = df_treino['preco'].clip(upper = 2500000) 

#------------------------------------------------------
# SELECIONAR COLUNAS
#------------------------------------------------------

colunas_selecionadas = [
     #'Id', 
     #'tipo', 
     #'bairro', 
     #'tipo_vendedor', 
     'quartos', 
     'suites', 
     'vagas',
     'area_util',
     'area_extra',
     #'diferenciais',
     'churrasqueira',
     'estacionamento',
     'piscina',
     'playground', 
     'quadra',
     's_festas',
     's_jogos',
     's_ginastica',
     'sauna',
     'vista_mar'
     ]

#------------------------------------------------------
# CONVERTER DATAFRAME PARA ARRAY NUMERICO
#------------------------------------------------------

X = df_treino[colunas_selecionadas].values
y = df_treino[['preco']].values

#------------------------------------------------------
# SEPARAR O CONJUNTO DE VALIDACAO
#------------------------------------------------------

from sklearn.model_selection import train_test_split

X_treino, X_valid, y_treino, y_valid = train_test_split(
    X,
    y,
    test_size=1000#,
    #random_state=428
    )

#------------------------------------------------------
# AJUSTAR TODAS AS VARIAVEIS NA ESCALA PADRAO
#------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler   = scaler.fit(X_treino)
X_treino = scaler.transform(X_treino)
X_valid  = scaler.transform(X_valid)

#------------------------------------------------------
# MODELO KNN
#------------------------------------------------------

import numpy as np
from sklearn.metrics   import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

modelo = KNeighborsRegressor(
    n_neighbors = 3,
    weights = 'uniform'    #'distance'
    )

modelo.fit(X_treino,y_treino)
y_pred = modelo.predict(X_valid)

print ( "RMSE =", np.sqrt(mean_squared_error(y_valid,y_pred)))














