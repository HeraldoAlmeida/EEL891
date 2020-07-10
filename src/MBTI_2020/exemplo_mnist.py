import numpy  as np
import pandas as pd

training_dataset = pd.read_csv("mnist_test.csv.zip")
test_dataset     = pd.read_csv("mnist_train.csv.zip")

X_train = training_dataset.iloc[:,1:].to_numpy()
X_test = test_dataset.iloc[:,1:].to_numpy()

Y_train = training_dataset.iloc[:,0].to_numpy()
Y_test = test_dataset.iloc[:,0].to_numpy()

X_train = X_train[]:100,:]
X_test  = X_test[:100,:]



import matplotlib.pyplot as plt

for sample in range(0,10):

    plt.figure(figsize=(4,4))
    d_plot = plt.subplot(1,1,1)

    d_plot.set_title("y = %.2f" % Y_test[sample])
     
    d_plot.imshow(X_test[sample,:].reshape(28,28),
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0,
                  vmax=255
                  )

    plt.show()
    

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

print ( '    k     Accuracy')
print ( ' ----     --------')

for k in range(1,2,2):
    
    knn = KNeighborsClassifier( n_neighbors = k )

    knn = knn.fit(X_train, Y_train)
    
    Y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score ( Y_test  , Y_pred  )

    print ( ('   %2d' % k) , ( '%10.2f' % (100*accuracy)  ) )


