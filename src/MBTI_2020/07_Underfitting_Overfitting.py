#==============================================================================
#  Underfitting e Overfitting
#==============================================================================

polynomial_degree = 9  # grau do polinomio usado no modelo

number_of_samples = 20  # numero de amostras de dados disponiveis

#------------------------------------------------------------------------------
#  Definir a função real (sem ruido) de onde vieram as amostras
#  (nao utilizada pelo regressor, usada somente para visualizacao grafica)
#------------------------------------------------------------------------------

import numpy as np

X_grid = np.linspace(0, 1.00, 101).reshape(-1,1)
y_grid = np.sin(2 * np.pi * X_grid)

#------------------------------------------------------------------------------
#  Gerar um conjunto de amostras com ruido gaussiano em torno da funcao
#------------------------------------------------------------------------------

np.random.seed(seed=0)

X_rand = np.random.rand(number_of_samples,1)
y_rand = np.sin(2 * np.pi * X_rand) + 0.20 * np.random.randn(number_of_samples,1)

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X_rand, 
        y_rand,
        test_size = 0.5 #, random_state = 352019
)

#------------------------------------------------------------------------------
#  Visualizar as amostras em um diagrama de dispersao
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Amostras Disponiveis')

plt.scatter ( X_train , y_train , 
              color = 'red'   , marker = 'o' , s = 30 , alpha = 0.5 ,
              label = 'Amostras para Treinamento'  
            )

plt.scatter ( X_test  , y_test  ,
              color = 'green' , marker = 'o' , s = 30 , alpha = 0.5 ,
              label = 'Amostras para Teste'        
            )

plt.plot    ( X_grid  , y_grid  ,
              color = 'grey'  , linestyle='dotted' ,
              label = 'Funcao alvo (desconhecida)' 
            )

plt.legend()

plt.xlabel('X')
plt.ylabel('y')

plt.ylim(-1.5,1.5)

plt.show()

#raise SystemExit()

#------------------------------------------------------------------------------
#  Treinar um regressor polinomial com o conjunto de treinamento
#------------------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pf = PolynomialFeatures(polynomial_degree)
modelo = LinearRegression()

Z_train = pf.fit_transform(X_train)

modelo = modelo.fit(Z_train, y_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o proprio conjunto de treinamento
#------------------------------------------------------------------------------

y_train_pred = modelo.predict(Z_train)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o conjunto de teste
#------------------------------------------------------------------------------

Z_test      = pf.transform(X_test)

y_test_pred = modelo.predict(Z_test)

#------------------------------------------------------------------------------
#  Calcular o desempenho do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )

R2_in  = r2_score ( y_train , y_train_pred )
R2_out = r2_score ( y_test  , y_test_pred  )

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o grid (para posteriormente plotar a curva)
#------------------------------------------------------------------------------

X_grid_poly = pf.transform(X_grid)
y_grid_pred = modelo.predict(X_grid_poly)

#------------------------------------------------------------------------------
#  Visualizar a resposta do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

fig = plt.figure ( figsize=(14,6) )

pin = plt.subplot ( "121" )
plt.ylim ( -1.5 , 1.5 )

pout = plt.subplot ( "122" )
plt.ylim ( -1.5 , 1.5 )

# Quadro com resultado DENTRO da amostra (conjunto de treinamento)

pin.title.set_text ( 'Aproximacao de grau '+ str(polynomial_degree) + 
                     '\nDesempenho DENTRO da amostra:'              +
                     '\n  R = ' +  str ( '%.4f' % R2_in  )           +
                     ' RMSE = ' + str ( '%.4f' % RMSE_in)
                   )

pin.plot    ( X_grid, y_grid     ,
              color = 'grey' , linestyle = 'dotted'  , alpha = 0.5 ,
              label='Funcao alvo (desconhecida)'
             )

pin.scatter ( X_train , y_train      ,  
              color = 'red'  , marker = 'o' , s = 60 , alpha = 0.5 ,
              label = 'Conjunto de Treinamento'
            )

pin.scatter ( X_train , y_train_pred ,
              color = 'blue' , marker = 'x' , s = 120 , alpha = 0.5 ,
              label = 'Respostas do Modelo'
            )

# pin.plot    ( X_grid, y_grid_pred,
#               color = 'blue' , linestyle = 'solid'   , alpha = 0.25,
#               label='Funcao correspondente ao modelo'
#             )


# Quadro com resultado FORA da amostra (conjunto de teste)

pout.title.set_text ( 'Aproximacao de grau '+ str(polynomial_degree) + 
                      '\nDesempenho FORA da amostra:'                +
                      '\n R = ' +  str ( '%.4f' % R2_out  )          +
                      ' RMSE = ' + str ( '%.4f' % RMSE_out)
                    )

pout.plot    ( X_grid , y_grid     ,
               color = 'grey'  , linestyle = 'dashed'  , alpha = 0.5,
               label = 'Funcao alvo (desconhecida)'
             )

pout.scatter ( X_test , y_test      ,
               color = 'green' , marker = 'o' , s = 60 , alpha = 0.5 ,
               label = 'Conjunto de Teste'
             )

pout.scatter ( X_test , y_test_pred ,
               color = 'blue'  , marker = 'x' , s =120 , alpha = 0.5 ,
               label = 'Respostas do Modelo'
             )

pout.plot    ( X_grid , y_grid_pred ,
               color = 'blue'  , linestyle = 'solid'   , alpha = 0.25,
               label='Funcao correspondente ao modelo'
             )

plt.show()
raise SystemExit()

#------------------------------------------------------------------------------
#  Calcular o RMSE para um modelo "perfeito" (clarividente)
#------------------------------------------------------------------------------

RMSE_in  = math.sqrt ( mean_squared_error ( y_train , np.sin(2 * np.pi * X_train) ) )
RMSE_out = math.sqrt ( mean_squared_error ( y_test  , np.sin(2 * np.pi * X_test)  ) )

R2_in  = r2_score ( y_train , y_train_pred )
R2_out = r2_score ( y_test  , y_test_pred  )


#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#------------------------------------------------------------------------------

print('\nParametros do regressor:\n', 
      np.append( lr.intercept_ , lr.coef_  ) )
#raise SystemExit()

#------------------------------------------------------------------------------
#  Exibir os coeficientes do polinomio
#------------------------------------------------------------------------------

#print ( ' Grau     Erro IN    Erro OUT')
#print ( ' ----     -------    --------')
#
#for degree in range(1,21):
#    
#    pf = PolynomialFeatures(degree)
#    lr = LinearRegression()
#
#    X_train_poly = pf.fit_transform(X_train)
#    lr = lr.fit(X_train_poly, y_train)
#    
#    y_train_pred = lr.predict(X_train_poly)
#    
#    X_test_poly = pf.transform(X_test)
#    y_test_pred = lr.predict(X_test_poly)
#    
#    RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
#    RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )
#
#    print ( str ( '   %2d' % degree   ) + '  ' +  
#            str ( '%10.4f' % RMSE_in  ) + '  ' +
#            str ( '%10.4f' % RMSE_out )
#          )


