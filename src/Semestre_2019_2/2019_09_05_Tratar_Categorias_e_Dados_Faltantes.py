#==============================================================================
#  Tratamento de Dados Faltantes e de Atributos Categóricos
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataset = pd.read_csv('../../data/D03_Categorias_e_Dados_Faltantes.csv')

#------------------------------------------------------------------------------
#  Separar os datasets em  atributos e alvo
#------------------------------------------------------------------------------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#------------------------------------------------------------------------------
#  Tratar os dados faltantes, preenchendo-os com valores estimados
#------------------------------------------------------------------------------

from sklearn.preprocessing import Imputer

# instanciar um Imputer para substituir todas as ocorrências de 'NaN'
# pelo valor médio (ou pela mediana, ou pelo valor mais frequente,
# conforme parâmetro "strategy") da linha ou da coluna (conforme
# parâmetro "axis")

imputer = Imputer(
        missing_values = 'NaN',  # lista de valores a serem substituidos
        strategy = 'median',       # pode ser também 'median' ou 'most_frequent'
        axis = 0                 # 0 para coluna, 1 para linha
)

# o método "fit" ajusta os parâmetros internos do Imputer, 
# conforme estratégia escolhida
 
imputer = imputer.fit(X[:, 1:3])

# o método transform preenche os dados faltantes com os valores 
# determinados pela estratégia escolhida

Xold = X.copy();

X[:, 1:3] = imputer.transform(X[:, 1:3])

#------------------------------------------------------------------------------
#  Codificar o atributo categórico da coluna 0 (paí­s)
#------------------------------------------------------------------------------

Xold = X.copy();

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# codifica os países da coluna 0 em rótulos numéricos 0, 1, 2, etc.

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Transforma a coluna 0 em um conjunto de colunas con conteúdo binário
# (uma coluna para cada valor distinto)

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X);
X = X.toarray();

#------------------------------------------------------------------------------
#  Codificar o alvo ('yes' ou 'no' - comprou ou não comprou)
#------------------------------------------------------------------------------

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


