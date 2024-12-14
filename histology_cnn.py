# Import necessary libraries
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from skimage.transform import resize
from tensorflow.image import resize_with_pad
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from numpy import flipud, rot90

# Project selection
project = "histology" # @param ["Choose your dataset!", "bees", "histology", "beans", "malaria"]

# URL dictionaries for the projects
article_url_dict = {
    "beans": "https://docs.google.com/document/d/19AcNUO-9F4E9Jtc4bvFslGhyuM5pLxjCqKYV3rUaiCc/edit?usp=sharing",
    "malaria": "https://docs.google.com/document/d/1u_iX2oDrEZ1clhFefpP3V8uwAjf7EUV4G6kq_3JDcVY/edit?usp=sharing",
    "histology": "https://docs.google.com/document/d/162WhUE9KqCgq_I7-VvENZD2n1IVsbeXVRSwfJEkxAqQ/edit?usp=sharing",
    "bees": "https://docs.google.com/document/d/1PUB_JuYHi6zyHsWAhkIb7D7ExeB1EfI09arc6Ad1bUY/edit?usp=sharing"
}

dataset_url_prefix_dict = {
    "histology": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/",
    "bees": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Safeguarding%20Bee%20Health/"
}

# Load dataset from tensorflow databank
if project == "Choose your dataset!":
    print("Please choose your dataset from the dropdown menu!")

elif project == "beans":
    data, info = tfds.load('beans', split='train[:1024]', as_supervised=True, with_info=True)
    feature_dict = info.features['label'].names
    images = np.array([resize_with_pad(image, 128, 128, antialias=True) for image, _ in data]).astype(int)
    labels = [feature_dict[int(label)] for _, label in data]

elif project == "malaria":
    data, info = tfds.load('malaria', split='train[:1024]', as_supervised=True, with_info=True)
    images = np.array([resize_with_pad(image, 256, 256, antialias=True) for image, _ in data]).astype(np.uint8)
    labels = ['malaria' if label == 1 else 'healthy' for _, label in data]

else:  # For histology and bees datasets
    wget_command = f'wget -q --show-progress "{dataset_url_prefix_dict[project]}'
    !{wget_command + 'images.npy" '}
    !{wget_command + 'labels.npy" '}

    images = np.load("images.npy")
    labels = np.load("labels.npy")

    !rm images.npy labels.npy

# One-hot encoding labels
y = np.array(pd.get_dummies(labels))
X = images

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Initialize CNN model
cnn_model = Sequential()
cnn_model.add(Input(shape=X_train.shape[1:]))
cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(len(set(labels)), activation='softmax'))

# Compile CNN model
cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Create a dictionary to map one-hot encoding back to class labels
one_hot_encoding_to_label_dict = {np.argmax(ohe): label for ohe, label in zip(y, labels)}

def ScoreVectorToPredictions(prob_vector):
    class_num = np.argmax(prob_vector)
    class_name = one_hot_encoding_to_label_dict[class_num]
    return class_name, max(prob_vector)

# Augment data with flip and rotation
X_train_augment, y_train_augment = [], []
for i in range(100):
    new_X = flipud(X_train[i])  # Flip image vertically
    new_y = y_train[i]
    X_train_augment.append(new_X)
    y_train_augment.append(new_y)

X_train_augment = np.array(X_train_augment)
y_train_augment = np.array(y_train_augment)

# Train the model with augmented data
cnn_model.fit(X_train_augment, y_train_augment, epochs=100, validation_data=(X_test, y_test))
