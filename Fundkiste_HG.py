import streamlit as st
from PIL import Image
# Falls hier noch ein Fehler kommt, wurde Punkt 1 noch nicht verarbeitet
from transformers import pipeline

st.title("🔍 Fundkiste Analyse (Lokal)")
st.write("KI-Analyse direkt auf dem Server – ohne Anmeldung.")

# Lädt das Modell herunter und speichert es im Cache
@st.cache_resource
def load_model():
    # Ein bewährtes Modell zur Bilderkennung von Google
    return pipeline("image-classification", model="google/vit-base-patch16-224")

# Zeigt eine Info während des Ladens
with st.spinner('KI-Modell wird vorbereitet... (Dauert beim ersten Mal kurz)'):
    classifier = load_model()

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Fix: Nutzt 'width' statt der veralteten Parameter [cite: 19, 21]
    st.image(image, caption='Hochgeladenes Bild', width='stretch')

    if st.button("Analyse starten"):
        with st.spinner('KI analysiert das Bild...'):
            results = classifier(image)
            
            st.success("Ergebnisse:")
            for prediction in results:
                label = prediction['label']
                score = round(prediction['score'] * 100, 2)
                st.info(f"**{label}** ({score}%)")
