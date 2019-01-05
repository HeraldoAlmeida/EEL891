#==============================================================================
#  Regressao usando KNN
#==============================================================================

import numpy as np
np.random.seed(seed=19532)
K = 5

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataset = pd.read_excel('../data/D11_Digits.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.ravel()

y = [ a%2 for a in y ]

#------------------------------------------------------------------------------
#  Visualizar alguns digitos
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

for i in range(0,5):
    plt.figure(figsize=(10,60))
    d_plot = plt.subplot(1, 5, i+1)
    d_plot.set_title("y = %.2f" % y[i])
 
    d_plot.imshow(X[i,:].reshape(8,8),
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0 , vmax=16)
    #plt.text(-8, 3, "y = %.2f" % y[i])

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
        test_size = 500 #, random_state = 352019
)

#------------------------------------------------------------------------------
# Aplicar transformacao de escala 
#------------------------------------------------------------------------------

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = StandardScaler()
##scaler = MinMaxScaler()
#
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)

#------------------------------------------------------------------------------
#  Treinar um regressor classificador kNN com o conjunto de treinamento
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

print ( 'Matriz de confusao DENTRO da amostra: ' )
print ( confusion_matrix(y_train, y_train_pred) )

print ( 'Matriz de confusao  FORA  da amostra: ' )
print ( confusion_matrix(y_test, y_test_pred) )

print ( 'Accuracy = %f %%' % (100*accuracy_score(y_test,y_test_pred)) )
#raise SystemExit

##------------------------------------------------------------------------------
##  Verificar erro DENTRO e FORA da amostra em funcao de K
##------------------------------------------------------------------------------

print ( '    k     Acc. IN    Acc. OUT')
print ( ' ----     -------    --------')

for k in range(1,21,2):
    
    lr = KNeighborsClassifier( n_neighbors = k )

    lr = lr.fit(X_train, y_train)
    
    y_train_pred = lr.predict(X_train)
    
    y_test_pred = lr.predict(X_test)
    
    acc_in  = accuracy_score ( y_train , y_train_pred )
    acc_out = accuracy_score ( y_test  , y_test_pred  )

    print ( str ( '   %2d' % k   ) + '  ' +  
            str ( '%10.4f' % acc_in  ) + '  ' +
            str ( '%10.4f' % acc_out )
          )

##------------------------------------------------------------------------------
## Regressao Logistica
##------------------------------------------------------------------------------
#
#from sklearn.linear_model import LogisticRegression
#
#print ( '    C     Acc. IN    Acc. OUT')
#print ( ' ----     -------    --------')
#
#for k in range(-6,6):
#    
#    c = 10**k
#    
#    lr = LogisticRegression(C = c, penalty='l2')
#
#    lr = lr.fit(X_train, y_train)
#    
#    y_train_pred = lr.predict(X_train)
#    
#    y_test_pred = lr.predict(X_test)
#    
#    acc_in  = accuracy_score ( y_train , y_train_pred )
#    acc_out = accuracy_score ( y_test  , y_test_pred  )
#
#    print ( str ( '   %2d' % k   ) + '  ' +  
#            str ( '%10.4f' % acc_in  ) + '  ' +
#            str ( '%10.4f' % acc_out )
#          )
#
#
###------------------------------------------------------------------------------
### Regressao Logistica
###------------------------------------------------------------------------------
##
##from sklearn.naive_bayes import GaussianNB, MultinomialNB
##
##print ( '    A     Acc. IN    Acc. OUT')
##print ( ' ----     -------    --------')
##
##for k in range(-3,9):
##    
##    a = 10**k
##
##    nb = MultinomialNB(alpha=a)
##    
##    nb = nb.fit(X_train, y_train)
##    
##    y_train_pred = nb.predict(X_train)
##    
##    y_test_pred = nb.predict(X_test)
##    
##    acc_in  = accuracy_score ( y_train , y_train_pred )
##    acc_out = accuracy_score ( y_test  , y_test_pred  )
##    
##    print ( str ( '   %2d' % k       ) + '  ' +  
##            str ( '%10.4f' % acc_in  ) + '  ' +
##            str ( '%10.4f' % acc_out )
##          )
##
