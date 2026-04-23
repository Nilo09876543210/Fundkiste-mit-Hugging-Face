import streamlit as st
from PIL import Image
from transformers import pipeline

# 1. Seite einrichten
st.set_page_config(page_title="Fundkiste", layout="centered")
st.title("🔍 Die einfache Fundkiste")
st.write("Lade ein Bild hoch und die KI sagt dir, was es ist.")

# 2. KI-Modell laden (Passiert lokal auf dem Server)
@st.cache_resource
def model_laden():
    # Wir nehmen ein Standard-Modell von Google
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner('KI wird gestartet...'):
    classifier = model_laden()

# 3. Bild-Upload
bild_datei = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "png", "jpeg"])

if bild_datei is not None:
    # Bild anzeigen
    bild = Image.open(bild_datei)
    st.image(bild, caption="Dein Bild", width='stretch') # Fix für veraltete Befehle

    # Analyse-Button
    if st.button("Was ist das?"):
        with st.spinner('KI denkt nach...'):
            ergebnisse = classifier(bild)
            
            st.success("Ich habe folgende Vermutungen:")
            for info in ergebnisse:
                name = info['label']
                sicherheit = round(info['score'] * 100, 1)
                st.write(f"**{name}** (Sicherheit: {sicherheit}%)")
