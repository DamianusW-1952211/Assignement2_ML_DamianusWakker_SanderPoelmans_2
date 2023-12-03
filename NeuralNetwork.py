import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# plt.imshow(x_train[0])

y_train = y_train.reshape(-1, )
# classes = [""]

x_train = x_train / 255
x_test = x_test / 255

# convolutional neural network
cnn = models.Sequential([
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),


    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(x_train, y_train, epochs=40)
cnn.evaluate(x_test, y_test)
y_pred = cnn.predict(x_test)
y_pred[:5]