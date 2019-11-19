#==============================================================================
#  Classificacao usando KNN
#==============================================================================

import numpy as np
np.random.seed(seed=19532)
K = 1

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataset = pd.read_excel('../../data/D11_Digits.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values.ravel()

#X = dataset.iloc[:, 0:8].values

#y = [ a%2 for a in y ]
#y = [ np.int(a==5) for a in y ]

#------------------------------------------------------------------------------
#  Visualizar alguns digitos
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

for i in range(0,10):
    plt.figure(figsize=(10,60))
    d_plot = plt.subplot(1, 10, i+1)
    d_plot.set_title("y = %.2f" % y[i+1000])
 
    d_plot.imshow(X[i+1000,:].reshape(8,8),
                  #interpolation='spline16',
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
#scaler = MinMaxScaler()

#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)

#------------------------------------------------------------------------------
#  Treinar um classificador LinearSVC com o conjunto de treinamento
#------------------------------------------------------------------------------

from sklearn.svm import LinearSVC

clf = LinearSVC(
        penalty='l2',
        C=10,
        multi_class='ovr'  # 'multinomial'
        )
clf = clf.fit(X_train, y_train)

#from sklearn.tree import DecisionTreeClassifier

#lr = DecisionTreeClassifier( )
#lr = lr.fit(X_train, y_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o proprio conjunto de treinamento
#------------------------------------------------------------------------------

y_train_pred = clf.predict(X_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o conjunto de teste
#------------------------------------------------------------------------------

y_test_pred = clf.predict(X_test)

#------------------------------------------------------------------------------
#  Calcular o desempenho do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#print ( 'C = %f' % C )
#print ( ' ' )

print ( 'Matriz de confusao DENTRO da amostra: ' )
print ( confusion_matrix(y_train, y_train_pred) )

print ( 'Matriz de confusao  FORA  da amostra: ' )
print ( confusion_matrix(y_test, y_test_pred) )

print ( 'Accuracy    = ' + str(100*accuracy_score(y_test,y_test_pred)) )
print ( 'Precision   = ' + str(100*precision_score(y_test,y_test_pred,average=None)) )
print ( 'Sensitivity = ' + str(100*recall_score(y_test,y_test_pred,average=None)) )
print ( 'F1          = ' + str(100*f1_score(y_test,y_test_pred,average=None)) )

##------------------------------------------------------------------------------
##  Verificar erro DENTRO e FORA da amostra em funcao de K
##------------------------------------------------------------------------------
#
#print ( '    k     Acc. IN    Acc. OUT')
#print ( ' ----     -------    --------')
#
#for k in range(1,41,1):
#    
#    lr = KNeighborsClassifier( n_neighbors = k )
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
#            )


#mistakes = [ 18, 118, 266]
#for i in range(3):
#    plt.figure(figsize=(3,60))
#    d_plot = plt.subplot(1, 3, i+1)
#    d_plot.set_title("y = %.2f" % y_test[mistakes[i]])
# 
#    d_plot.imshow(X_test[mistakes[i],:].reshape(8,8),
#                  #interpolation='spline16',
#                  interpolation='nearest',
#                  cmap='binary',
#                  vmin=0 , vmax=1)
#    #plt.text(-8, 3, "y = %.2f" % y[i])
#
#    d_plot.set_xticks(())
#    d_plot.set_yticks(())
# 
#plt.show()


###------------------------------------------------------------------------------
### SUPPORT VECTOR MACHINE LINEAR
###------------------------------------------------------------------------------

from sklearn.svm import LinearSVC

print ( '    m           C     Acc. IN    Acc. OUT')
print ( ' ----     -------     -------    --------')

for k in range(-250,-199,5):
    
    m = k/100
    c = 10**m
    
    LinSVC = LinearSVC(penalty='l2', C=c)

    LinSVC = LinSVC.fit(X_train, y_train)

    y_train_pred = LinSVC.predict(X_train)
    y_test_pred  = LinSVC.predict(X_test)

    acc_in  = accuracy_score ( y_train , y_train_pred )
    acc_out = accuracy_score ( y_test  , y_test_pred  )


    print ( str ( ' %4.2f' % m       ) + '  ' +  
            str ( '%10.4f' % c       ) + '  ' +  
            str ( '%10.4f' % acc_in  ) + '  ' +
            str ( '%10.4f' % acc_out )
          )


###------------------------------------------------------------------------------
### SUPPORT VECTOR MACHINE COM KERNEL
##------------------------------------------------------------------------------
#
#from sklearn.svm import SVC
#
#print ( '    m           C     Acc. IN    Acc. OUT')
#print ( ' ----     -------     -------    --------')
#
#for k in range(-3,4,1):
#    
#    m = k
#    c = 10**m
#    g = 10**m
#    
#    rbfSVC = SVC(kernel='rbf', gamma=10**(-2.7), C=c)
#
#    rbfSVC = rbfSVC.fit(X_train, y_train)
#
#    y_train_pred = rbfSVC.predict(X_train)
#    y_test_pred  = rbfSVC.predict(X_test)
#
#    acc_in  = accuracy_score ( y_train , y_train_pred )
#    acc_out = accuracy_score ( y_test  , y_test_pred  )
#
#    print ( str ( ' %4.1f' % m       ) + '  ' +  
#            str ( '%10.4f' % c       ) + '  ' +  
#            str ( '%10.4f' % acc_in  ) + '  ' +
#            str ( '%10.4f' % acc_out )
#          )
#
#
#
#
###------------------------------------------------------------------------------
###  Verificar erro DENTRO e FORA da amostra em funcao de K
###------------------------------------------------------------------------------
#
##print ( '    k     Acc. IN    Acc. OUT')
##print ( ' ----     -------    --------')
##
##for k in range(1,41,2):
##    
##    lr = KNeighborsClassifier( n_neighbors = k )
##
##    lr = lr.fit(X_train, y_train)
##    
##    y_train_pred = lr.predict(X_train)
##    
##    y_test_pred = lr.predict(X_test)
##    
##    acc_in  = accuracy_score ( y_train , y_train_pred )
##    acc_out = accuracy_score ( y_test  , y_test_pred  )
##
##    print ( str ( '   %2d' % k   ) + '  ' +  
##            str ( '%10.4f' % acc_in  ) + '  ' +
##            str ( '%10.4f' % acc_out )
##          )
##
###------------------------------------------------------------------------------
### Regressao Logistica
###------------------------------------------------------------------------------
##
##from sklearn.linear_model import LogisticRegression
##
##print ( '    C     Acc. IN    Acc. OUT')
##print ( ' ----     -------    --------')
##
##for k in range(-6,7):
##    
##    c = 10**k
##    
##    lr = LogisticRegression(C = c, penalty='l2')
##
##    lr = lr.fit(X_train, y_train)
##    
##    y_train_pred = lr.predict(X_train)
##    
##    y_test_pred = lr.predict(X_test)
##    
##    acc_in  = accuracy_score ( y_train , y_train_pred )
##    acc_out = accuracy_score ( y_test  , y_test_pred  )
##
##    print ( str ( '   %2d' % k   ) + '  ' +  
##            str ( '%10.4f' % acc_in  ) + '  ' +
##            str ( '%10.4f' % acc_out )
##          )
##
##
####------------------------------------------------------------------------------
#### Classificador Bayesiano Ingenuo
####------------------------------------------------------------------------------
###
###from sklearn.naive_bayes import GaussianNB, MultinomialNB
###
###print ( '    A     Acc. IN    Acc. OUT')
###print ( ' ----     -------    --------')
###
###for k in range(-3,9):
###    
###    a = 10**k
###
###    nb = MultinomialNB(alpha=a)
###    
###    nb = nb.fit(X_train, y_train)
###    
###    y_train_pred = nb.predict(X_train)
###    
###    y_test_pred = nb.predict(X_test)
###    
###    acc_in  = accuracy_score ( y_train , y_train_pred )
###    acc_out = accuracy_score ( y_test  , y_test_pred  )
###    
###    print ( str ( '   %2d' % k       ) + '  ' +  
###            str ( '%10.4f' % acc_in  ) + '  ' +
###            str ( '%10.4f' % acc_out )
###          )
###
