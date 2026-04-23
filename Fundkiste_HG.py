import streamlit as st
import requests
from PIL import Image
import io

# 1. Konfiguration
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
headers = {"Authorization": f"Bearer {'DEIN_HUGGING_FACE_TOKEN'}"}

st.title("👕 Schul-Fundkiste KI")
st.write("Lade ein Foto hoch, um zu sehen, was es ist!")

# 2. Upload-Bereich
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Dein Foto', use_column_width=True)
    
    # Bild für die API vorbereiten
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    if st.button('KI fragen...'):
        with st.spinner('Die KI überlegt...'):
            response = requests.post(API_URL, headers=headers, data=byte_im)
            results = response.json()
            
            # Ergebnis anzeigen
            st.success("Ich bin mir ziemlich sicher:")
            # Das erste Ergebnis ist das mit der höchsten Wahrscheinlichkeit
            label = results[0]['label']
            score = round(results[0]['score'] * 100, 1)
            st.metric(label=label, value=f"{score}% sicher")
