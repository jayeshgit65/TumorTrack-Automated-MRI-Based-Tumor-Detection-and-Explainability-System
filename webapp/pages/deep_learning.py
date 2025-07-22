import sys
import os

# Add the project root directory (or src directory) to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from glob import glob
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras
from PIL import Image
import cv2
import base64

# Page Configuration 
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.markdown("""
    <style>
    .reportview-container { background: #f5f5f5; }
    .stButton > button { width: 100%; border-radius: 10px; }
    .stImage { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# Make sure the model path is correct relative to this file
model = load_model(r"C:\Users\lenovo\Documents\Projects\-TumorVision-MLOps-Pipeline-Automated-MRI-Based-Detection-and-Explainability-main\models\trained.h5")

# Class Labels
classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

def preprocess_image(image):
    # CHANGED: Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # CHANGED: Convert PIL image to numpy array
    image = np.array(image)
    # CHANGED: Resize the image to (255, 255)
    image = tf.image.resize(image, [255, 255])
    # CHANGED: Normalize pixel values to [0, 1]
    image = image / 255.0
    # CHANGED: Add a batch dimension so that shape becomes (1, 255, 255, 3)
    image = np.expand_dims(image, axis=0)
    return image

def generate_gradcam(img_tensor, model, last_conv_layer_name='block5_conv3'):
    # Create a model that maps the input to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]
    # Compute gradients of the top predicted class with respect to conv outputs
    grads = tape.gradient(pred_output, conv_outputs)
    # Pool the gradients over all axes except channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Multiply each feature map by the pooled gradient
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (255, 255))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

tumor_info = {
    "no_tumor": {
        "description": "No signs of tumor detected in the MRI scan.",
        "symptoms": "None",
        "growth_rate": "N/A",
        "severity": "None"
    },
    "pituitary_tumor": {
        "description": "Tumor in the pituitary gland, located at the base of the brain. May affect hormone production.",
        "symptoms": "Hormonal Imbalance, Vision Problems, Headaches",
        "growth_rate": "Slow",
        "severity": "Medium"
    },
    "meningioma_tumor": {
        "description": "Tumor that arises from the meninges, the membranes that surround the brain and spinal cord.",
        "symptoms": "Headaches, Seizures, Vision Issues, Personality Changes",
        "growth_rate": "Slow to Moderate",
        "severity": "Medium to High"
    },
    "glioma_tumor": {
        "description": "Tumor originating in the glial cells of the brain or spine. Can be aggressive.",
        "symptoms": "Seizures, Headaches, Nausea, Cognitive Decline",
        "growth_rate": "Fast",
        "severity": "High"
    }
}

def download_report(pred_class, confidence):
    """ Generate and return a downloadable medical report """
    info = tumor_info[pred_class]
    report_text = f"""\
Brain Tumor Classification Report
----------------------------------
Prediction : {pred_class}
Confidence : {confidence:.2f}%

----------------------------------

Description:
{info['description']}

Common Symptoms:
{info['symptoms']}

Growth Rate:
{info['growth_rate']}

Severity:
{info['severity']}
"""

    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📄 Download Medical Report</a>'
    return href


# Sidebar for Image Upload
st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]

        # Display Result
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        with col2:
            # gradcam_image = generate_gradcam(image, model)
            # st.image(gradcam_image, caption="Grad-CAM Visualisation", use_column_width=True)

            heatmap = generate_gradcam(input_tensor, model)
            # Overlay heatmap on original image
            img = np.array(image.resize((255,255)))
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
        
        # Download Report
        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)


