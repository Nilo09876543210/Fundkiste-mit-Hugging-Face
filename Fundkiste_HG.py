import streamlit as st
from PIL import Image
# Diese Zeile verursacht aktuell den Fehler, bis Punkt 1 erledigt ist
from transformers import pipeline

st.title("🔍 Fundkiste Analyse (Lokal)")
st.write("KI-Analyse direkt auf dem Server – ohne Anmeldung.")

# Lädt das Modell herunter und speichert es im Zwischenspeicher
@st.cache_resource
def load_model():
    # 'base' ist ein solides Standardmodell zur Bilderkennung
    return pipeline("image-classification", model="google/vit-base-patch16-224")

# Zeigt eine Lade-Info beim ersten Mal
with st.spinner('KI-Modell wird vorbereitet... (Dauert beim ersten Start kurz)'):
    classifier = load_model()

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild', width='stretch') # [cite: 22]

    if st.button("Analyse starten"):
        with st.spinner('KI analysiert das Bild...'):
            results = classifier(image)
            
            st.success("Ergebnisse:")
            for prediction in results:
                label = prediction['label']
                score = round(prediction['score'] * 100, 2)
                st.info(f"**{label}** ({score}%)")
