import pandas as pd
import matplotlib.pyplot as plt

training_dataset = pd.read_csv("mnist_train.csv.zip")
test_dataset     = pd.read_csv("mnist_test.csv.zip")

training_dataset = training_dataset.iloc[:60000,:]
test_dataset     = test_dataset.iloc[:10000,:]

X_train = training_dataset.iloc[:,1:].to_numpy()
X_test  = test_dataset.iloc[:,1:].to_numpy()

Y_train = training_dataset.iloc[:,0].to_numpy().ravel()
Y_test  = test_dataset.iloc[:,0].to_numpy().ravel()


    

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(
    n_neighbors = 3,
    weights = 'distance',
    n_jobs = -1
    )

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators      = 500,
    criterion         = 'gini',
    max_depth         = None,
    min_samples_split =  2,
    min_samples_leaf  =  1,
    max_features      = 'sqrt',
    oob_score         = False,
    n_jobs            = -1
    )

classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
Y_pred = Y_pred.ravel()

for sample in range(0,10):

    plt.figure(figsize=(4,4))
    d_plot = plt.subplot(1,1,1)

    d_plot.set_title(
        str("y_true = %.0f" % Y_test[sample]) + " ; " +
        str("y_pred = %.0f" % Y_pred[sample])
        )
     
    d_plot.imshow(X_test[sample,:].reshape(28,28),
                  interpolation='nearest',
                  cmap='binary',
                  vmin=0,
                  vmax=255
                  )

    plt.show()


accuracy = accuracy_score(Y_test,Y_pred)

print ( "Accuracy (%%): %.2f" % (100*accuracy) )

print(confusion_matrix(Y_test,Y_pred))







