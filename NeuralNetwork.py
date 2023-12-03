import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess the training dataset
train_data = pd.read_csv('Training_set.csv')
train_images = []
train_labels = []

for index, row in train_data.iterrows():
    img = load_img(row['filename'], color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    train_images.append(img_array)
    train_labels.append(row['label'])

train_images = np.array(train_images).astype('float32') / 255

# Use LabelEncoder for label encoding
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_encoded, test_size=0.2, random_state=42)

# Load and preprocess the test dataset
test_data = pd.read_csv('Testing_set.csv')
test_images = []

for index, row in test_data.iterrows():
    img = load_img(row['filename'], color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    test_images.append(img_array)

test_images = np.array(test_images).astype('float32') / 255

# Create a simple convolutional neural network (CNN)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the test set
y_pred = np.argmax(model.predict(test_images), axis=1)
predicted_labels = label_encoder.inverse_transform(y_pred)

# Compare predicted labels with actual labels
test_data['predicted_label'] = predicted_labels
test_data.to_csv('Predictions.csv', index=False)

