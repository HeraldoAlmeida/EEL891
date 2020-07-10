import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist

( x_train , y_train ) , ( x_test , y_test ) = mnist.load_data()


print(x_train[0])

for i in range(5):
    plt.imshow ( x_train[i] , cmap = plt.cm.binary )
    plt.show()

x_train = tf.keras.utils.normalize ( x_train , axis = 1 )

print(x_train[0])


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,tf.nn.relu))
model.add(tf.keras.layers.Dense(128,tf.nn.relu))
model.add(tf.keras.layers.Dense(10,tf.nn.softmax))

model.compile (
    
    optimizer = "adam",
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
    
    )

model.fit (
    
    x_train,
    y_train,
    epochs = 3
    
    )

val_loss , val_acc = model.evaluate(x_test,y_test)
print ( "Loss = " , val_loss , " ; Accuracy = " , val_acc )

predictions = model.predict(x_test)
y_pred = np.argmax(predictions,axis=1)

print ( "Accuracy = ", 100 * accuracy_score(y_test,y_pred) )
