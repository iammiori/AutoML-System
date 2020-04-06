import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential #tensorflow2
from tensorflow.keras.layers import Dense #tensorflow2

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

Data_set = np.loadtxt("https://raw.githubusercontent.com/gilbutITbook/006958/master/deeplearning/dataset/pima-indians-diabetes.csv",delimiter=",")

X = Data_set[:,0:8]
Y = Data_set[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))