import streamlit as st
from PIL import Image
from transformers import pipeline

# Titel
st.title("🔍 Fundkiste Analyse (Lokal)")
st.write("Diese App nutzt KI direkt im Browser - ohne Anmeldung!")

# Modell laden (wird beim ersten Mal heruntergeladen)
@st.cache_resource
def load_model():
    # Ein kleines, schnelles Modell, das keine Anmeldung braucht
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# Datei-Uploader
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild', width='stretch')

    if st.button("Analyse starten"):
        with st.spinner('KI arbeitet lokal...'):
            # Analyse direkt im Code
            results = classifier(image)
            
            st.success("Ergebnis:")
            for prediction in results:
                label = prediction['label']
                score = round(prediction['score'] * 100, 2)
                st.info(f"**{label}** ({score}%)")
