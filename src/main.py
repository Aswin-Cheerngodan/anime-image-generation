import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained generator model
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model('models/DCGEN (2).h5')

generator = load_generator()
latent_dim = 100  # Update if your DCGAN used a different dimension

st.title("🧠 DCGAN Image Generator")
st.write("Click the button to generate a new image from random noise using your trained DCGAN.")

# Predict button
if st.button("Predict"):
    # Generate random noise
    noise = tf.random.normal([1, 300])

    # Generate image
    generated_image = generator(noise, training=False)

    # Rescale from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) / 2.0
    generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)
    generated_image = tf.squeeze(generated_image).numpy()

    # If image is grayscale (single channel), handle accordingly
    if len(generated_image.shape) == 2:
        image = Image.fromarray((generated_image * 255).astype(np.uint8), mode='L')
    else:
        image = Image.fromarray((generated_image * 255).astype(np.uint8))

    st.image(image, caption="Generated Image", use_container_width=True)
