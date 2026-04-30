import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Fundkiste KI", 
    page_icon="🔍", 
    layout="centered"
)

# Custom CSS für das Styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar
with st.sidebar:
    st.title("Über die Fundkiste")
    st.info("Diese App nutzt KI, um Objekte in Bildern zu erkennen.")

# 3. Hauptinhalt
st.title("🔍 Fundkiste: Bild-Analyse")

try:
    from transformers import pipeline
    system_bereit = True
except ImportError:
    system_bereit = False

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

if not system_bereit:
    st.error("KI-Module werden noch geladen...")
    st.stop()

classifier = load_model()

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_container_width=True)
    
    with col2:
        if st.button("🚀 Analysieren"):
            with st.spinner("KI denkt nach..."):
                results = classifier(image)
                for res in results:
                    st.write(f"**{res['label']}**")
                    st.progress(res['score'])
else:
    st.info("Bitte lade ein Bild hoch.")
