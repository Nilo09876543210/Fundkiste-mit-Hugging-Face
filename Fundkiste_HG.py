import streamlit as st
import requests
from PIL import Image
import io

# Titel und Layout
st.set_page_config(page_title="Fundkiste KI", layout="centered")
st.title("🔍 Fundkiste Bild-Analyse")
st.write("Lade ein Bild hoch, um zu sehen, was die KI darin erkennt.")

# Datei-Uploader
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Fix 1: Einrückung korrigiert 
    image = Image.open(uploaded_file)
    
    # Fix 2: Modernes Layout ohne Deprecation Warnings [cite: 19, 21]
    st.image(image, caption='Dein hochgeladenes Bild', width='stretch')

    # API Konfiguration (ViT Modell als Standard)
    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    
    # Fix 3: Zugriff auf Secrets (verhindert den KeyError) [cite: 7]
    try:
        headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    except Exception:
        st.error("🔑 Fehler: Das 'HF_TOKEN' wurde in den Streamlit-Secrets nicht gefunden!")
        st.stop()

    if st.button("Analyse starten"):
        with st.spinner('KI analysiert das Bild...'):
            try:
                # Bild für API vorbereiten
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_bytes = img_byte_arr.getvalue()

                # API Anfrage
                response = requests.post(API_URL, headers=headers, data=img_bytes)
                
                # Fix 4: JSON-Fehler abfangen [cite: 7, 10, 13]
                if response.status_code == 200:
                    results = response.json()
                    st.success("Ergebnis der Analyse:")
                    for prediction in results:
                        label = prediction.get('label', 'Unbekannt')
                        score = round(prediction.get('score', 0) * 100, 2)
                        st.info(f"**{label}** (Wahrscheinlichkeit: {score}%)")
                elif response.status_code == 503:
                    st.warning("Das Modell wird gerade geladen. Bitte in 30 Sekunden noch einmal versuchen.")
                else:
                    st.error(f"API Fehler {response.status_code}: {response.text}")
            
            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
