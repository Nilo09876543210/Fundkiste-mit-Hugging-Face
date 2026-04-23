import streamlit as st
import requests
from PIL import Image
import io

# --- KONFIGURATION ---
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"

# Token-Check (Streamlit Secrets oder direkt)
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
else:
    hf_token = "DEIN_HF_TOKEN_HIER" # Ersetze dies, falls du keine Secrets nutzt

headers = {"Authorization": f"Bearer {hf_token}"}

# --- UI ---
st.title("👕 Die KI-Fundkiste")
st.write("Lade ein Foto hoch, um das Kleidungsstück zu bestimmen.")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # WICHTIG: Alles hier drunter muss genau gleich weit eingerückt sein!
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)
    
    # Bild konvertieren
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    if st.button('KI Analyse starten'):
        with st.spinner('Warte auf Antwort von Hugging Face...'):
            response = requests.post(API_URL, headers=headers, data=byte_im)
            
            if response.status_code == 200:
                results = response.json()
                label = results[0]['label']
                score = round(results[0]['score'] * 100, 1)
                st.success(f"Gefunden: {label} ({score}% sicher)")
            elif response.status_code == 503:
                st.warning("Modell lädt noch... bitte in 10 Sekunden nochmal drücken.")
            else:
                st.error(f"Fehler {response.status_code}: {response.text}")
