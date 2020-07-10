#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados IRIS (problema de classificacao)
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Iris em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../../data/D01_iris.xlsx')

#------------------------------------------------------------------------------
#  Separar em dataframes distintos os atributos e o alvo 
#    - os atributos são todas as colunas menos a última
#    - o alvo é a última coluna 
#------------------------------------------------------------------------------

attributes = dataframe.iloc[:,:-1]
target     = dataframe.iloc[:,4]

#------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = attributes.to_numpy()
y = target.to_numpy()

#------------------------------------------------------------------------------
#  Visualizar a mariz de dispersão dos 4 atributos
#------------------------------------------------------------------------------

from matplotlib.colors import ListedColormap

foo = pd.plotting.scatter_matrix(
        attributes, 
        c=y, 
        figsize=(11, 11),
        marker='o',
        hist_kwds={'bins': 40},
        s=30, 
        alpha=0.5,
        #cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        cmap=ListedColormap(['red', 'green', 'blue'])
)

#------------------------------------------------------------------------------
# Visualizar um gráfico de dispersão 3D para um subconjunto de 3 atributos
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# preapara lista com os nomes das colunas (rótulos dos eixos do gráfico)

columnNames = list(attributes)

# cria figura

myFigure = plt.figure(figsize=(15,12))

# cria os eixos do primeiro subplot em um quadro 1x1 de subplots na figura

myAxes = myFigure.add_subplot(111, projection='3d')

# estes são os atributos que serão plotados nos eixos X, Y e Z
# (identificados pelos números das colunas)

colx, coly, colz = 0,2,3

xs = X[:,colx]
ys = X[:,coly]
zs = X[:,colz]

myAxes.scatter(xs, ys, zs, c=y, marker='o', s=40, alpha=0.5, 
           cmap=ListedColormap(['red', 'green' , 'blue']) )

myAxes.set_xlabel(columnNames[colx])
myAxes.set_ylabel(columnNames[coly])
myAxes.set_zlabel(columnNames[colz])

plt.show()

#------------------------------------------------------------------------------
# Dividir os dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=50
    #random_state=20200702
    )

#------------------------------------------------------------------------------
# Treinar um classificador KNN para identificar a especie
#------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

print("K  Accuracy")
print("-- --------")

for k in range(1,21):
    
    knn_classifier = KNeighborsClassifier(
        n_neighbors = k,
        weights = 'uniform'
        )
    
    knn_classifier.fit(X_train,y_train)
    y_pred = knn_classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    
    # cm = confusion_matrix(y_test,y_pred)    
    # print("Confusion Matrix =")
    # print(cm)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    print ( "%2d" % k , "%.2f %%" % (100*accuracy))

#y_pred vs y_test