#==============================================================================
#  Regressao usando KNN
#==============================================================================

import numpy as np
np.random.seed(seed=19532)
K = 1

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataset = pd.read_excel('../data/D01_iris.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

#X = dataset.iloc[:, :-1].values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.ravel()

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 50 #, random_state = 352019
)

#------------------------------------------------------------------------------
# Aplicar transformacao de escala 
#------------------------------------------------------------------------------

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)

#------------------------------------------------------------------------------
#  Treinar um regressor polinomial com o conjunto de treinamento
#------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier( n_neighbors = K )
knn = knn.fit(X_train, y_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o proprio conjunto de treinamento
#------------------------------------------------------------------------------

y_train_pred = knn.predict(X_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o conjunto de teste
#------------------------------------------------------------------------------

y_test_pred = knn.predict(X_test)

#------------------------------------------------------------------------------
#  Calcular o desempenho do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print ( 'K = %d' % K )
print ( ' ' )

print ( 'Matriz de confusao DENTRO da amostra: ' )
print ( confusion_matrix(y_train, y_train_pred) )

print ( 'Matriz de confusao  FORA  da amostra: ' )
print ( confusion_matrix(y_test, y_test_pred) )


#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#------------------------------------------------------------------------------

print ( '    k     Accu IN    Accu OUT')
print ( ' ----     -------    --------')

for k in range(1,21):
    
    knn = KNeighborsClassifier( n_neighbors = k )
    knn = knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred  = knn.predict(X_test)
    
    #error_in  = precision_score ( y_train , y_train_pred , average = 'macro' )
    #error_out = precision_score ( y_test  , y_test_pred  , average = 'macro' )

    error_in  = accuracy_score ( y_train , y_train_pred )
    error_out = accuracy_score ( y_test  , y_test_pred  )

    print ( str ( '   %2d' % k   ) + '  ' +  
            str ( '%10.4f' % error_in  ) + '  ' +
            str ( '%10.4f' % error_out )
          )



