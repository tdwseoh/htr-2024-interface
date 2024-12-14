import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from skimage.transform import resize
from tensorflow.image import resize_with_pad
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import requests
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from numpy import flipud
import streamlit as st
import speech_recognition as sr

st.title("Image Classification with CNN")

# ==================== Speech Recognition =====================

st.header("Speech Recognition")
st.write("Click on the button below and speak to convert speech to text.")

# Button to start speech recognition
if st.button("Start Listening"):
    recognizer = sr.Recognizer()

    # Displaying the listening status
    st.write("Listening... Speak now!")

    with sr.Microphone() as source:
        try:
            # Adjust for ambient noise and listen for the audio input
            recognizer.adjust_for_ambient_noise(source) 
            audio = recognizer.listen(source)

            # Recognize speech using Google's Web Speech API
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized Text: {text}")
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand the speech.")
        except sr.RequestError as e:
            st.write(f"Error with speech recognition service: {e}")

# ==================== Image Upload Section =====================

st.header("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Open the uploaded image using PIL
image = Image.open(uploaded_file).convert('RGB')
st.image(image, caption="Uploaded Image", use_column_width=True)

# =================== Helper Functions =====================

def download_file(url, filename):
    """Download a file from a URL and save it locally."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        raise Exception(f"Failed to download {filename}. Status code: {response.status_code}")

# ================= Project and Dataset =====================

project = "histology"  # Options: "beans", "malaria", "histology", "bees"

dataset_url_prefix_dict = {
    "histology": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/",
    "bees": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Safeguarding%20Bee%20Health/"
}

if project == "beans":
    data, info = tfds.load('beans', split='train[:1024]', as_supervised=True, with_info=True)
    feature_dict = info.features['label'].names
    images = np.array([resize_with_pad(image, 128, 128, antialias=True) for image, _ in data]).astype(int)
    labels = [feature_dict[int(label)] for _, label in data]

elif project == "malaria":
    data, info = tfds.load('malaria', split='train[:1024]', as_supervised=True, with_info=True)
    images = np.array([resize_with_pad(image, 256, 256, antialias=True) for image, _ in data]).astype(np.uint8)
    labels = ['malaria' if label == 1 else 'healthy' for _, label in data]

else:  # For histology and bees datasets
    # Define file URLs
    image_url = f"{dataset_url_prefix_dict[project]}images.npy"
    labels_url = f"{dataset_url_prefix_dict[project]}labels.npy"

    # Download files
    download_file(image_url, "images.npy")
    download_file(labels_url, "labels.npy")

    # Load the downloaded data
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    # Clean up: remove files after loading
    os.remove("images.npy")
    os.remove("labels.npy")

# ==================== Data Preprocessing ====================

# One-hot encoding labels
y = np.array(pd.get_dummies(labels))
X = images

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== CNN Model =====================

cnn_model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(32, (3, 3), activation='relu', padding="same"),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding="same"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding="same"),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding="same"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding="same"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Data Augmentation
X_train_augment = np.array([flipud(img) for img in X_train])
y_train_augment = y_train

X_train_final = np.concatenate((X_train, X_train_augment), axis=0)
y_train_final = np.concatenate((y_train, y_train_augment), axis=0)

# Train the model
cnn_model.fit(X_train_final, y_train_final, epochs=35, validation_data=(X_test, y_test))

# Map predictions back to class names
one_hot_to_label = {i: label for i, label in enumerate(pd.get_dummies(labels).columns)}

# ==================== Streamlit App =====================

if uploaded_file is not None:
    
    # Display a spinner while the AI is processing the image
    with st.spinner("AI is analyzing the image..."):
        # Preprocess the image
        input_image = np.array(image)
        input_image_resized = resize(input_image, X_train.shape[1:], anti_aliasing=True)  # Resize to CNN input shape
        input_image_resized = np.expand_dims(input_image_resized, axis=0)  # Add batch dimension

        # Make prediction
        predictions = cnn_model.predict(input_image_resized)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]
        class_label = one_hot_to_label[class_index]

    # Display prediction and confidence after analysis
    st.subheader(f"Prediction Result:")
    st.write(f"Predicted Class: {class_label}")
    st.write(f"Confidence: {confidence:.2f}")
