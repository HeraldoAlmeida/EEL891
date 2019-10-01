#==============================================================================
#  Regularizacao Ridge e Lasso
#==============================================================================
import numpy as np

np.random.seed(seed=1960532)

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_excel('../../data/D02_Boston.xlsx')

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 0.5 #, random_state = 2222
)

X_train_original = X_train

#------------------------------------------------------------------------------
#  Aplicar uma escala a matriz X
#------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler,MinMaxScaler

StdSc   = StandardScaler()
X_train = StdSc.fit_transform(X_train)
X_test  = StdSc.transform(X_test)

#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra 
#  para regressor K-NN em funcao do parametro K
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor

print ( ' K-NN' )
print ( ' ' )
print ( '   K      Erro IN    Erro OUT' )
print ( ' ----     -------    --------' )

for k in range(1,21):

    knn = KNeighborsRegressor(n_neighbors=k,weights='distance',p=1)
    knn = knn.fit(X_train, y_train)
    
    y_train_pred_knn = knn.predict(X_train)
    y_test_pred_knn  = knn.predict(X_test)
    
    RMSE_in_knn  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_knn ) )
    RMSE_out_knn = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_knn  ) )
    
    print ( str ( '   %2d' % k            ) + '  ' +  
            str ( '%10.4f' % RMSE_in_knn  ) + '  ' +
            str ( '%10.4f' % RMSE_out_knn )
          )


#
##print ( '--------------------------------------------------------------------')
##print ( ' LINEAR SVM' )
##print ( '--------------------------------------------------------------------')
##print ( ' ' )
##
##from sklearn.svm import LinearSVR
##
##print ( '    C     Acc. IN    Acc. OUT')
##print ( ' ----     -------    --------')
##
##for k in range(-6,6):
##    
##    c = 10**k
##    
##    classifier = LinearSVR(C = c)
##
##    classifier = classifier.fit(X_train, y_train.ravel())
##    
##    y_train_pred = classifier.predict(X_train)
##    
##    y_test_pred = classifier.predict(X_test)
##    
###    y_train_pred_knn = knn.predict(X_train)
###    y_test_pred_knn  = knn.predict(X_test)
##    
##    RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
##    RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )
##    
##    print ( str ( '   %2d' % k            ) + '  ' +  
##            str ( '%10.4f' % RMSE_in  ) + '  ' +
##            str ( '%10.4f' % RMSE_out )
##          )
##
