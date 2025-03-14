import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import shutil

# Set dataset path
data_path = "C:/Users/kalpana/OneDrive/Desktop/Kalpana project/Dataset"
train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'val')

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar Menu
menu = st.sidebar.selectbox("Menu", ["Data", "EDA - Visual", "Prediction"])

# Model file
model_file = "cnn_model.h5"

# Image parameters
img_size = (128, 128)

# Class labels
classes = ['ClassA', 'ClassB']

# Helper function to load and preprocess image
def preprocess_image(image_file):
    image = load_img(image_file, target_size=img_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Load or Train model
def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save(model_file)
    return model, history, val_generator

# Load model or train if not exists
if not os.path.exists(model_file):
    with st.spinner("Training model... Please wait"):
        model, history, val_generator = train_model()
        st.session_state.model_trained = True
else:
    model = load_model(model_file)
    st.session_state.model_trained = True

# Menu 1 - DATA
if menu == "Data":
    st.header("Dataset Overview")
    if os.path.exists(train_path):
        for class_name in os.listdir(train_path):
            class_dir = os.path.join(train_path, class_name)
            images = os.listdir(class_dir)
            st.subheader(f"Class: {class_name} - {len(images)} images")
            cols = st.columns(5)
            for i, image in enumerate(images[:5]):
                img_path = os.path.join(class_dir, image)
                cols[i % 5].image(img_path, width=100)

    st.subheader("Model Performance Metrics")
    if os.path.exists(model_file):
        if 'history' not in locals():
            _, history, val_generator = train_model()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        fig1 = px.line(x=list(range(1, len(acc)+1)), y=acc, labels={'x':'Epoch', 'y':'Accuracy'}, title='Training Accuracy')
        fig2 = px.line(x=list(range(1, len(val_acc)+1)), y=val_acc, labels={'x':'Epoch', 'y':'Validation Accuracy'}, title='Validation Accuracy')
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        st.subheader("Confusion Matrix")
        y_pred = model.predict(val_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = val_generator.classes
        cm = confusion_matrix(y_true, y_pred_classes)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        fig_cm = px.imshow(df_cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig_cm)

# Menu 2 - EDA Visual
elif menu == "EDA - Visual":
    st.header("Exploratory Data Analysis")
    class_counts = {}
    for class_name in os.listdir(train_path):
        class_counts[class_name] = len(os.listdir(os.path.join(train_path, class_name)))

    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Image Count'])
    fig = px.bar(df, x='Class', y='Image Count', title='Image Count per Class', color='Class')
    st.plotly_chart(fig)

    st.subheader("Image Dimension Analysis")
    dims = []
    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        for img_file in os.listdir(class_dir)[:30]:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            dims.append(img.shape[:2])
    dim_df = pd.DataFrame(dims, columns=['Height', 'Width'])
    fig2 = px.histogram(dim_df, x='Height', title="Height Distribution")
    fig3 = px.histogram(dim_df, x='Width', title="Width Distribution")
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

# Menu 3 - Prediction
elif menu == "Prediction":
    st.header("Image Prediction")
    uploaded_file = st.file_uploader("Upload an image to classify", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            image = preprocess_image(uploaded_file)
            prediction = model.predict(image)
            class_idx = np.argmax(prediction)
            class_name = classes[class_idx]
            confidence = prediction[0][class_idx] * 100
            st.success(f"Predicted Class: {class_name} ({confidence:.2f}% confidence)")

