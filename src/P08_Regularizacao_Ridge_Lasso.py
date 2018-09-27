#==============================================================================
#  Regularizacao Ridge e Lasso
#==============================================================================

polynomial_degree =    7  # grau do polinomio usado no modelo

number_of_samples =   200  # numero de amostras de dados disponiveis

#------------------------------------------------------------------------------
#  Definir a função real (sem ruido) de onde vieram as amostras
#  (nao utilizada pelo regressor, usada somente para visualizacao grafica)
#------------------------------------------------------------------------------

import numpy as np

X_grid = np.linspace(0, 1.0, 100).reshape(-1,1)
y_grid = np.sin(2 * np.pi * X_grid)

#------------------------------------------------------------------------------
#  Gerar um conjunto de amostras com ruido gaussiano em torno da funcao
#------------------------------------------------------------------------------

np.random.seed(seed=0)

#np.random.seed(seed=19670532)

X_rand = np.random.rand(number_of_samples,1)
y_rand = np.sin(2 * np.pi * X_rand)  + 0.1 * np.random.randn(number_of_samples,1)

#------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X_rand, 
        y_rand,
        test_size = 0.5 #, random_state = 2222
)

#------------------------------------------------------------------------------
#  Visualizar as amostras em um diagrama de dispersao
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

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

#------------------------------------------------------------------------------
#  Treinar um regressor polinomial padrao com o conjunto de treinamento
#------------------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pf = PolynomialFeatures(polynomial_degree)
X_train_poly = pf.fit_transform(X_train)


lr = LinearRegression()
lr = lr.fit(X_train_poly, y_train)

#------------------------------------------------------------------------------
#  Treinar regressores polinomiais com regularizacao ridge e lasso 
#------------------------------------------------------------------------------

from sklearn.linear_model import Ridge, Lasso

# Regressor com regularizacao Ridge

lr_ridge = Ridge ( alpha = 1.E-3 )
lr_ridge = lr_ridge.fit ( X_train_poly , y_train )

# Regressor com regularizacao lasso

lr_lasso = Lasso ( alpha = 1.E-12 )
lr_lasso = lr_lasso.fit ( X_train_poly , y_train )

#------------------------------------------------------------------------------
#  Obter a resposta dos modelos para o conjunto de treinamento
#------------------------------------------------------------------------------

y_train_pred       =       lr.predict(X_train_poly)
y_train_pred_ridge = lr_ridge.predict(X_train_poly)
y_train_pred_lasso = lr_lasso.predict(X_train_poly)

#------------------------------------------------------------------------------
#  Obter a resposta do modelo padrao e dos modelos regularizados 
#  para o conjunto de teste
#------------------------------------------------------------------------------

X_test_poly  = pf.transform(X_test)

y_test_pred       =       lr.predict(X_test_poly)
y_test_pred_ridge = lr_ridge.predict(X_test_poly)
y_test_pred_lasso = lr_lasso.predict(X_test_poly)

#------------------------------------------------------------------------------
#  Calcular o desempenho do modelo dentro e fora da amostra
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

# desempenho do regressor original

RMSE_in  = math.sqrt ( mean_squared_error ( y_train , y_train_pred ) )
RMSE_out = math.sqrt ( mean_squared_error ( y_test  , y_test_pred  ) )

R2_in  = r2_score ( y_train , y_train_pred )
R2_out = r2_score ( y_test  , y_test_pred  )

# desempenho do regressor com regularizacao ridge

RMSE_in_ridge  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_ridge ) )
RMSE_out_ridge = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_ridge  ) )

R2_in_ridge  = r2_score ( y_train , y_train_pred_ridge )
R2_out_ridge = r2_score ( y_test  , y_test_pred_ridge  )

# desempenho do regressor com regularizacao lasso

RMSE_in_lasso  = math.sqrt ( mean_squared_error ( y_train , y_train_pred_lasso ) )
RMSE_out_lasso = math.sqrt ( mean_squared_error ( y_test  , y_test_pred_lasso  ) )

R2_in_lasso  = r2_score ( y_train , y_train_pred_lasso )
R2_out_lasso = r2_score ( y_test  , y_test_pred_lasso  )

#------------------------------------------------------------------------------
#  Obter a resposta do modelo para o grid (para posteriormente plotar a curva)
#------------------------------------------------------------------------------

X_grid_poly = pf.transform(X_grid)

y_grid_pred       =       lr.predict(X_grid_poly)
y_grid_pred_ridge = lr_ridge.predict(X_grid_poly)
y_grid_pred_lasso = lr_lasso.predict(X_grid_poly)

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
                     '\n R = ' +  str ( '%.4f' % R2_in  )           +
                     ' RMSE = ' + str ( '%.4f' % RMSE_in)
                   )

pin.plot    ( X_grid, y_grid     ,
              color = 'grey' , linestyle = 'dashed'  , alpha = 0.5 ,
              label='Funcao alvo (desconhecida)'
             )

pin.scatter ( X_train , y_train      ,  
              color = 'red'  , marker = 'o' , s = 30 , alpha = 0.5 ,
              label = 'Conjunto de Treinamento'
            )

#pin.scatter ( X_train , y_train_pred ,
#              color = 'blue' , marker = 'x' , s = 60 , alpha = 0.5 ,
#              label = 'Respostas do Modelo'
#            )

pin.plot    ( X_grid, y_grid_pred,
              color = 'blue' , linestyle = 'solid'   , alpha = 0.25,
              label='Funcao correspondente ao modelo padrao'
            )

pin.plot    ( X_grid, y_grid_pred_ridge,
              color = 'magenta' , linestyle = 'solid'   , alpha = 0.25,
              label='Funcao correspondente ao modelo ridge'
            )

pin.plot    ( X_grid, y_grid_pred_lasso,
              color = 'orange' , linestyle = 'solid'   , alpha = 0.25,
              label='Funcao correspondente ao modelo LASSO'
            )


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
               color = 'green' , marker = 'o' , s = 30 , alpha = 0.5 ,
               label = 'Conjunto de Teste'
             )

#pout.scatter ( X_test , y_test_pred ,
#               color = 'blue'  , marker = 'x' , s = 60 , alpha = 0.5 ,
#               label = 'Respostas do Modelo'
#             )

pout.plot    ( X_grid , y_grid_pred ,
               color = 'blue'  , linestyle = 'solid'   , alpha = 0.25,
               label='Funcao correspondente ao modelo'
             )

pout.plot    ( X_grid, y_grid_pred_ridge,
               color = 'magenta' , linestyle = 'solid'   , alpha = 0.25,
               label='Funcao correspondente ao modelo ridge'
             )

pout.plot    ( X_grid, y_grid_pred_lasso,
               color = 'orange' , linestyle = 'solid'   , alpha = 0.25,
               label='Funcao correspondente ao modelo LASSO'
             )


plt.show()

print(lr.coef_)
print(lr.intercept_)

#------------------------------------------------------------------------------
#  Verificar erro DENTRO e FORA da amostra em funcao do grau do polinomio
#------------------------------------------------------------------------------

print ( '           Regressao Simples   |     Regressao RIDGE    |     Regressao LASSO  ' )
print ( ' ' )
print ( ' Grau     Erro IN    Erro OUT  |   Erro IN    Erro OUT  |   Erro IN    Erro OUT' )
print ( ' ----     -------    --------  |   -------    --------  |   -------    --------' )

for degree in range(1,21):
    
    pf = PolynomialFeatures(degree)
    lr = LinearRegression()

    X_train_poly = pf.fit_transform(X_train)
    lr = lr.fit(X_train_poly, y_train)

    lr_ridge = lr_ridge.fit ( X_train_poly , y_train )
    lr_lasso = lr_lasso.fit ( X_train_poly , y_train )
    
    y_train_pred = lr.predict(X_train_poly)
    
    y_train_pred_ridge = lr_ridge.predict(X_train_poly)
    y_train_pred_lasso = lr_lasso.predict(X_train_poly)

    X_test_poly = pf.transform(X_test)
    y_test_pred = lr.predict(X_test_poly)
    
    y_test_pred_ridge = lr_ridge.predict(X_test_poly)
    y_test_pred_lasso = lr_lasso.predict(X_test_poly)

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


