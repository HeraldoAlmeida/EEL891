#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados DIGITS (problema de classificacao)
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Digits em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../../data/D11_Digits.xlsx')

#------------------------------------------------------------------------------
#  Separar em dataframes distintos os atributos e o alvo 
#    - os atributos são todas as colunas menos a última
#    - o alvo é a última coluna 
#------------------------------------------------------------------------------

attributes = dataframe.iloc[:,1:-1]
target     = dataframe.iloc[:,-1]

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = attributes.to_numpy()
y = target.to_numpy()

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

for sample in range(0,10):

    plt.figure(figsize=(4,4))
    d_plot = plt.subplot(1,1,1)

    d_plot.set_title("y = %.2f" % y[sample])
     
    d_plot.imshow(X[sample,:].reshape(8,8),
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0,
                  vmax=16
                  )

    plt.show()


#------------------------------------------------------------------------------
# Dividir os dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=500,
    random_state=20200702
    )

#------------------------------------------------------------------------------
# Treinar uma rede neural 
#------------------------------------------------------------------------------

import numpy
import tensorflow

from sklearn.metrics import accuracy_score


z_train = numpy.zeros((y_train.shape[0],10))

for i in range(y_train.shape[0]):
    z_train[i,y_train[i]] = 1


tf_model = tensorflow.keras.models.Sequential()

tf_model.add(tensorflow.keras.layers.Flatten(input_shape=(64,)))
tf_model.add(tensorflow.keras.layers.Dense(4096,tensorflow.nn.relu))
tf_model.add(tensorflow.keras.layers.Dense(4096,tensorflow.nn.relu))
tf_model.add(tensorflow.keras.layers.Dense(10,tensorflow.nn.sigmoid))

tf_model.compile (
    
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
    
    )

tf_model.fit (
    
    X_train,
    z_train,
    epochs = 10
    )

z_pred = tf_model.predict(X_test)

y_pred = numpy.argmax(z_pred,axis=1)


print ( "TensorFlow   Accuracy = %5.2f %%" % ( 100 * accuracy_score(y_test,y_pred) ) )




#------------------------------------------------------------------------------
# Treinar um classificador KNN para identificar o digito
#------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(
    n_neighbors = 1,
    weights = 'uniform'
    )

from sklearn.tree import DecisionTreeClassifier

# knn_classifier = DecisionTreeClassifier()

#------------------------------------------------------------------------------
# Treinar o classificador e obter o resultado para o conjunto de teste
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score

knn_classifier.fit(X_train,y_train)
y_pred = knn_classifier.predict(X_test)

#------------------------------------------------------------------------------
# Mostrar a matriz de confusão e a acuracia
#------------------------------------------------------------------------------

cm = confusion_matrix(y_test,y_pred)    
print("Confusion Matrix =")
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy = %.1f %%" % (100*accuracy))


import matplotlib.pyplot as plt

for sample in range(0,4):

    plt.figure(figsize=(4,4))
    d_plot = plt.subplot(1,1,1)

    d_plot.set_title("y = %.2f %.2f" % ( y_test[sample] , y_pred[sample] ) )
     
    d_plot.imshow(X_test[sample,:].reshape(8,8),
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0,
                  vmax=16
                  )

    plt.show()


#------------------------------------------------------------------------------
# Explorar a variacao da acuracia com o parametro k
#------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

print("K  Accuracy")
print("-- --------")

for k in range(1,101):
    
    # classifier = KNeighborsClassifier(
    #     n_neighbors = k,
    #     weights = 'uniform'
    #     )

    # classifier = DecisionTreeClassifier(
    #     criterion='gini',
    #     max_features=None,
    #     max_depth=k
    #     )

    ne = k*5
    classifier = RandomForestClassifier(
        n_estimators=ne,
        max_features='auto'
        )
    
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    
    
    
    accuracy = accuracy_score(y_test,y_pred)
    
    print ( "%2d" % k , "%.2f %%" % (100*accuracy) , 'ne = %d'%ne )

#y_pred vs y_test