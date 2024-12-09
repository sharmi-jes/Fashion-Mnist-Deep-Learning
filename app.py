import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained Fashion MNIST model (Assumes you have a saved model)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_model.h5")

# Class labels for Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load the model
model = load_model()

# Streamlit app
st.title("Fashion MNIST Image Classification")

st.sidebar.title("Options")
options = st.sidebar.radio("Choose an option:", ["Upload Image", "View Sample Data"])

if options == "Upload Image":
    st.subheader("Upload a 28x28 grayscale image")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Process uploaded file
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image for the model
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = image_array.reshape(1, 28, 28, 1)  # Add batch dimension
        
        # Predict
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        
        # Show prediction
        st.write(f"Predicted Class: **{CLASS_NAMES[predicted_class]}**")
        st.bar_chart(predictions[0])  # Show prediction probabilities

elif options == "View Sample Data":
    st.subheader("Sample Fashion MNIST Data with Predictions")
    
    # Load Fashion MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    
    # Display a random sample from the dataset
    idx = np.random.choice(len(x_train), 9)
    sample_images = x_train[idx]
    sample_labels = y_train[idx]
    
    # Predict on sample images
    predictions = model.predict(sample_images.reshape(-1, 28, 28, 1) / 255.0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Plot the images with predictions
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Pred: {CLASS_NAMES[predicted_classes[i]]}")
    st.pyplot(fig)
