#==============================================================================
#  Classificacao usando KNN
#==============================================================================

import numpy as np
np.random.seed(seed=123456)
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

#from sklearn.neighbors import KNeighborsClassifier

#lr = KNeighborsClassifier( n_neighbors = K )
#lr = lr.fit(X_train, y_train)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for md in range(2,21):

    lr = DecisionTreeClassifier( max_features=64, max_depth=md)
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
    
#    
    print ( str ( 'MD = %2d' % md) + ' ' +
            str ( 'Accuracy = %f %%' % (100*accuracy_score(y_test,y_test_pred)) ) )
    
print(' ')
print(' BAGGING')
print(' ')


from sklearn.ensemble import BaggingClassifier


for ne in range(1,31,2):

    clf = BaggingClassifier(n_estimators=ne)
    clf = clf.fit(X_train, y_train)
    
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
        
    print ( str ( 'NE = %2d' % ne) + ' ' +
            str ( 'Accuracy = %f %%' % (100*accuracy_score(y_test,y_test_pred)) ) )
    

print(' ')
print(' RANDOM FOREST')
print(' ')


from sklearn.ensemble import RandomForestClassifier


for ne in range(10,201,10):

    clf = RandomForestClassifier(n_estimators=ne)
    clf = clf.fit(X_train, y_train)
    
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
        
    print ( str ( 'NE = %2d' % ne) + ' ' +
            str ( 'Accuracy = %f %%' % (100*accuracy_score(y_test,y_test_pred)) ) )

print(' ')
print(' EXTRA TREES')
print(' ')

    
from sklearn.ensemble import ExtraTreesClassifier

for ne in range(25,500,25):

    clf = ExtraTreesClassifier(n_estimators=ne)
    clf = clf.fit(X_train, y_train)
    
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
        
    print ( str ( 'NE = %2d' % ne) + ' ' +
            str ( 'Accuracy = %f %%' % (100*accuracy_score(y_test,y_test_pred)) ) )
    






