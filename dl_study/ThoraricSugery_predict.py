import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential #tensorflow2
from tensorflow.keras.layers import Dense #tensorflow2

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

Data_set = np.loadtxt("https://raw.githubusercontent.com/gilbutITbook/006958/master/deeplearning/dataset/ThoraricSurgery.csv",delimiter=",")

X = Data_set[:,0:17]
Y = Data_set[:,17]

model =tf.keras.Sequential() #tf2
model.add(Dense(30, input_dim=17, activation='relu')) #tf2
model.add(Dense(1, activation='sigmoid')) #tf2

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

print("Accuracy:{0:0.4f}".format(model.evaluate(X,Y)[1]))