import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained generator model
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model('models\DCGEN.h5')

# Load generator
generator = load_generator()

# Correct latent dimension based on your model definition
latent_dim = 300

# Streamlit app UI
st.title("🎨 DCGAN Anime Image Generator")
st.write("Click the button to generate a new anime-style image using the trained DCGAN generator.")

# Predict button
if st.button("Generate Image"):
    images = []
    
    for i in range(16):
    # Generate random noise with correct shape
        noise = tf.random.normal([1, latent_dim])  # [batch_size, latent_dim]

        # Generate image using the model
        generated_image = generator(noise, training=False)

        # Remove batch dimension and convert to NumPy
        generated_image = tf.squeeze(generated_image).numpy()

        # Convert to PIL Image and display
        image = Image.fromarray((generated_image * 255).astype(np.uint8))
        images.append(image)

    # Number of images per row
    num_columns = 4

    # Loop through images in batches
    for i in range(0, len(images), num_columns):
        cols = st.columns(num_columns)
        for j, img in enumerate(images[i:i + num_columns]):  
            cols[j].image(img, caption=f"Image {i+j+1}", use_container_width=False, width=128)

