import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN",strategy="median",axis=0,verbose=0)
 
X_train = [ [ 1 ,   2] , [ np.nan , 3 ] , [ 7 , 6 ]]

X_test = [ [ np.nan , 2 ] , [ 6 , np.nan ] , [ np.nan , np.nan ]]
    
imp.fit(X_train)

X_trans = imp.transform(X_test)

print(X_test)
print(X_trans)
                
