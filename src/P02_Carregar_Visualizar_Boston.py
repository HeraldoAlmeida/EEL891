#==============================================================================
#  Carga e Visualizacao do Conjunto de Dados BOSTON (problema de regressão)
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados Boston em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd
dataframe = pd.read_excel('../data/D02_Boston.xlsx')

#------------------------------------------------------------------------------
#  Verificar os nomes das colunas disponíveis
#------------------------------------------------------------------------------

columnNames = dataframe.columns

print("Nomes das colunas:\n")
print(columnNames)

#------------------------------------------------------------------------------
#  Imprimir o gráfico de dispersão do alvo em relação ao atributo 'LSTAT'
#------------------------------------------------------------------------------

dataframe.plot.scatter(x='LSTAT',y='target')

#------------------------------------------------------------------------------
#  Imprimir os gráficos de dispersão do alvo em relação a acda atributo
#------------------------------------------------------------------------------

for col in columnNames:
    dataframe.plot.scatter(x=col,y='target')

