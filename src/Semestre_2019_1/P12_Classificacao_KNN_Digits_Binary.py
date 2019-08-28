#==============================================================================
#  Regressao usando KNN
#==============================================================================

import numpy as np

np.random.seed(seed=19532)
#np.random.seed(seed=156932)

K = 1

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_excel('../data/D11_Digits.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

#X = dataset.iloc[:, :-1].values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.ravel()


y = [ a%2 for a in y ]

#------------------------------------------------------------------------------
#  Visualizar alguns digitos
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

for i in range(0,5):
    d_plot = plt.subplot(1, 5, i+1)
    d_plot.set_title("y = %.2f" % y[i])
 
    d_plot.imshow(X[i,:].reshape(8,8), interpolation='nearest', cmap='binary', vmin=0 , vmax=16)
    #plt.text(-8, 3, "y = %.2f" % y[i])

    d_plot.set_xticks(())
    d_plot.set_yticks(())
    d_plot.set_xticks(())
    d_plot.set_yticks(())

plt.show()

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        train_size = 50 #, random_state = 352019
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

lr = KNeighborsClassifier( n_neighbors = K )
lr = lr.fit(X_train, y_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o proprio conjunto de treinamento
#------------------------------------------------------------------------------

y_train_pred = lr.predict(X_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o conjunto de teste
#------------------------------------------------------------------------------

y_test_pred = lr.predict(X_test)

#------------------------------------------------------------------------------
#  Calcular o desempenho do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print ( 'K = %d' % K )
print ( ' ' )

print ( 'Desempenho DENTRO da amostra: ' )

print ( confusion_matrix(y_train, y_train_pred) )

print ( "Accuracy  = %9.6f" %  accuracy_score(y_train, y_train_pred))
print ( "Precision = %9.6f" % precision_score(y_train, y_train_pred))
print ( "Recall    = %9.6f" %    recall_score(y_train, y_train_pred))
print ( "F1 Score  = %9.6f" %        f1_score(y_train, y_train_pred))

print ( 'Desempenho  FORA  da amostra: ' )

print ( confusion_matrix(y_test, y_test_pred) )

print ( "Accuracy  = %9.6f" %  accuracy_score(y_test, y_test_pred))
print ( "Precision = %9.6f" % precision_score(y_test, y_test_pred))
print ( "Recall    = %9.6f" %    recall_score(y_test, y_test_pred))
print ( "F1 Score  = %9.6f" %        f1_score(y_test, y_test_pred))


#print ( '   Erro DENTRO da amostra: ' confusion_matrix(y_train, y_train_pred) )
#print ( '   Erro  FORA  da amostra: ' confusion_matrix(y_test, y_test_pred) )




#from sklearn.metrics import confusion_matrix
#>>> y_true = [2, 0, 2, 2, 0, 1]
#>>> y_pred = [0, 0, 2, 2, 0, 2]
#>>> confusion_matrix(y_true, y_pred)
#array([[2, 0, 0],
#[0, 0, 1],
#[1, 0, 2]])




#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#------------------------------------------------------------------------------

#print ( '    k     Erro IN    Erro OUT')
#print ( ' ----     -------    --------')
#
#for k in range(1,31):
#    
#    lr = KNeighborsClassifier( n_neighbors = k )
#
#    lr = lr.fit(X_train, y_train)
#    
#    y_train_pred = lr.predict(X_train)
#    
#    y_test_pred = lr.predict(X_test)
#    
#    RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
#    RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )
#
#    print ( str ( '   %2d' % k   ) + '  ' +  
#            str ( '%10.4f' % RMSE_in  ) + '  ' +
#            str ( '%10.4f' % RMSE_out )
#          )
#
#
