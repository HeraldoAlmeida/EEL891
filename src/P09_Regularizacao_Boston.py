import numpy as np

np.random.seed(seed=1960532)

#==============================================================================
#  Regularizacao Ridge e Lasso
#==============================================================================

#polynomial_degree =    9  # grau do polinomio usado no modelo

#number_of_samples =   20  # numero de amostras de dados disponiveis

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_excel('../data/D02_Boston.xlsx')

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

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#
#X_train = scaler.fit_transform(X_train)
#X_test  = scaler.transform(X_test)
    

#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#  para regressores simples, ridge e lasso
#------------------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso


print ( '           Regressao Simples   |     Regressao RIDGE    |     Regressao LASSO  ' )
print ( ' ' )
print ( ' Grau     Erro IN    Erro OUT  |   Erro IN    Erro OUT  |   Erro IN    Erro OUT' )
print ( ' ----     -------    --------  |   -------    --------  |   -------    --------' )

for degree in range(1,7):
    
    
    # Transformar atributos originais em atributos polinomiais 

    pf = PolynomialFeatures(degree)
    X_train_poly = pf.fit_transform(X_train)
    X_test_poly  = pf.transform(X_test)

    #print(X_train_poly.shape)

    # Aplicar transformacao de escala nos atributos polinomiais 

    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly  = scaler.transform(X_test_poly)
    
    # Treinar regressores polinomiais 
    
    lr = LinearRegression()
    lr = lr.fit(X_train_poly, y_train)

    # Treinar regressor polinomial com regularizacao Ridge
    
    lr_ridge = Ridge ( alpha = 50 , max_iter=1000000 )  # 4.E+1 Boston ; 
    lr_ridge = lr_ridge.fit ( X_train_poly , y_train )
    
    # Treinar regressor polinomial com regularizacao lasso
    
    lr_lasso = Lasso ( alpha = 0.1 , max_iter=1000000 )  # 1.E-1 Boston ;
    lr_lasso = lr_lasso.fit ( X_train_poly , y_train )

    # Prever resposta ao conjunto de treinamento para os 3 regressores
    
    y_train_pred       = lr.predict(X_train_poly)
    y_train_pred_ridge = lr_ridge.predict(X_train_poly)
    y_train_pred_lasso = lr_lasso.predict(X_train_poly)

    # Prever resposta ao conjunto de teste para os 3 regressores
    
    y_test_pred       = lr.predict(X_test_poly)
    y_test_pred_ridge = lr_ridge.predict(X_test_poly)
    y_test_pred_lasso = lr_lasso.predict(X_test_poly)

    #  Calcular o desempenho do modelo dentro e fora da amostra

    import math
    from sklearn.metrics import mean_squared_error

    RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
    RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )

    RMSE_in_ridge  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_ridge ) )
    RMSE_out_ridge = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_ridge  ) )

    RMSE_in_lasso  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_lasso ) )
    RMSE_out_lasso = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_lasso  ) )

    print ( str ( '   %2d' % degree   ) + '  ' +  
            str ( '%10.4f' % RMSE_in  ) + '  ' +
            str ( '%10.4f' % RMSE_out ) + '  |' +
            str ( '%10.4f' % RMSE_in_ridge  ) + '  ' +
            str ( '%10.4f' % RMSE_out_ridge ) + '  |' +
            str ( '%10.4f' % RMSE_in_lasso  ) + '  ' +
            str ( '%10.4f' % RMSE_out_lasso )
          )


# Treinar regressor KNN 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

print ( ' K-NN' )
print ( ' ' )
print ( '   K      Erro IN    Erro OUT' )
print ( ' ----     -------    --------' )

for k in range(1,51,2):

    knn = KNeighborsRegressor()

#    StdSc   = StandardScaler()
#    X_train = StdSc.fit_transform(X_train)
#    X_test  = StdSc.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn = knn.fit(X_train, y_train)
    
    y_train_pred_knn = knn.predict(X_train)
    y_test_pred_knn  = knn.predict(X_test)
    
    RMSE_in_knn  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_knn ) )
    RMSE_out_knn = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_knn  ) )
    
    print ( str ( '   %2d' % k            ) + '  ' +  
            str ( '%10.4f' % RMSE_in_knn  ) + '  ' +
            str ( '%10.4f' % RMSE_out_knn )
          )

#print ( '--------------------------------------------------------------------')
#print ( ' LINEAR SVM' )
#print ( '--------------------------------------------------------------------')
#print ( ' ' )
#
#from sklearn.svm import LinearSVR
#
#print ( '    C     Acc. IN    Acc. OUT')
#print ( ' ----     -------    --------')
#
#for k in range(-6,6):
#    
#    c = 10**k
#    
#    classifier = LinearSVR(C = c)
#
#    classifier = classifier.fit(X_train, y_train.ravel())
#    
#    y_train_pred = classifier.predict(X_train)
#    
#    y_test_pred = classifier.predict(X_test)
#    
##    y_train_pred_knn = knn.predict(X_train)
##    y_test_pred_knn  = knn.predict(X_test)
#    
#    RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
#    RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )
#    
#    print ( str ( '   %2d' % k            ) + '  ' +  
#            str ( '%10.4f' % RMSE_in  ) + '  ' +
#            str ( '%10.4f' % RMSE_out )
#          )
#
