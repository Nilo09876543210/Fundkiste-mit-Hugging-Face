import streamlit as st
from PIL import Image
import numpy as np  # NumPy ist jetzt dabei!

st.title("Fundkiste mit NumPy & KI")

try:
    from transformers import pipeline
    st.success("✅ System bereit!")
except ImportError:
    st.error("❌ Die Bausteine fehlen noch. Bitte requirements.txt prüfen!")
    st.stop()

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # --- NUMPY INTEGRATION ---
    # Wir wandeln das Bild in ein NumPy-Array um
    bild_array = np.array(image)
    st.write(f"Bildgröße als Tabelle (NumPy): {bild_array.shape}")
    # --------------------------

    st.image(image, width='stretch')
    
    if st.button("Analyse starten"):
        results = classifier(image)
        st.write("Ergebnisse:", results)
