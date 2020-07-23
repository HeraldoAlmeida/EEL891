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

    plt.figure(figsize=(2,2))
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
# Importar classes de apuracao de metricas
#------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score

#------------------------------------------------------------------------------
# KNN
#------------------------------------------------------------------------------

print ( '+---------------------------------')
print ( '| KNN')
print ( '+---------------------------------')

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors = 1,
    weights = 'uniform'
    )

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)    
print("Confusion Matrix =")
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("KNN Accuracy = %.1f %%" % (100*accuracy))

#------------------------------------------------------------------------------
# RANDOM FOREST
#------------------------------------------------------------------------------

print ( '+---------------------------------')
print ( '| RANDOM FOREST')
print ( '+---------------------------------')

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,
    max_features='auto'
    )

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)    
print("Confusion Matrix =")
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print("RANDOM FOREST Accuracy = %.1f %%" % (100*accuracy))

#------------------------------------------------------------------------------
# REDE NEURAL (usando TensorFlow) 
#------------------------------------------------------------------------------

print ( '+---------------------------------')
print ( '| REDE NEURAL (usando TensorFlow)')
print ( '+---------------------------------')

import numpy
import tensorflow

#from tensorflow.keras.layers import Flatten,Dense 

z_train = numpy.zeros((y_train.shape[0],10))

for i in range(y_train.shape[0]):
    z_train[i,y_train[i]] = 1

tf_model = tensorflow.keras.models.Sequential()

# camada de entrada (1 neuronio para cada variavel de entrada)
tf_model.add(tensorflow.keras.layers.Flatten(input_shape=(64,)))

# camadas internas (ocultas) ---> aqui o projeto eh livre
tf_model.add(tensorflow.keras.layers.Dense(1000,tensorflow.nn.sigmoid))
tf_model.add(tensorflow.keras.layers.Dense(1000,tensorflow.nn.sigmoid))

# camda de saida (1 neuronio para cada classe)
tf_model.add(tensorflow.keras.layers.Dense(10,tensorflow.nn.sigmoid))

tf_model.compile (
    optimizer = "adam", #SGD RMSprop Adam Adadelta Adagrad Adamax Nadam Ftrl
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
    ) # vejam https://keras.io/api/models/model_training_apis/

tf_model.fit (   
    X_train,
    z_train,
    epochs = 10
    )

z_pred = tf_model.predict(X_test)

y_pred = numpy.argmax(z_pred,axis=1)


print ( "TensorFlow Accuracy = %5.2f %%" % ( 100 * accuracy_score(y_test,y_pred) ) )

