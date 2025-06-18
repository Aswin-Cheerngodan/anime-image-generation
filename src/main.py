import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained generator model
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model('models/DCGEN (2).h5')

# Load generator
generator = load_generator()

# Correct latent dimension based on your model definition
latent_dim = 300

# Streamlit app UI
st.title("🎨 DCGAN Anime Image Generator")
st.write("Click the button to generate a new anime-style image using the trained DCGAN generator.")

# Predict button
if st.button("Generate Image"):
    # Generate random noise with correct shape
    noise = tf.random.normal([1, latent_dim])  # [batch_size, latent_dim]

    # Generate image using the model
    generated_image = generator(noise, training=False)

    # Remove batch dimension and convert to NumPy
    generated_image = tf.squeeze(generated_image).numpy()

    # Convert to PIL Image and display
    
    image = Image.fromarray((generated_image * 255).astype(np.uint8))


    st.image(image, caption="🧠 Generated Image", use_container_width=True)
