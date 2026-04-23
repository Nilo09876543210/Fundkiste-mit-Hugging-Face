import streamlit as st
import requests
from PIL import Image

# Titel
st.title("Fundkiste Analyse")

# Datei-Uploader
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Fix 1: Korrekte Einrückung
    image = Image.open(uploaded_file)
    # Fix 2: Neuer Breiten-Parameter
    st.image(image, caption='Hochgeladenes Bild', width='stretch')

    # WICHTIG: Ersetze 'DEIN_MODELL' durch dein echtes Hugging Face Modell!
    API_URL = "https://api-inference.huggingface.co/models/DEIN_MODELL"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    if st.button("Analyse starten"):
        with st.spinner('KI arbeitet...'):
            response = requests.post(API_URL, headers=headers, data=uploaded_file.getvalue())
            
            # Fix 3: Erst Status prüfen, dann JSON laden
            if response.status_code == 200:
                try:
                    results = response.json()
                    st.success("Ergebnis:")
                    st.write(results)
                except:
                    st.error("Antwort konnte nicht gelesen werden.")
            elif response.status_code == 503:
                st.warning("Das Modell lädt noch. Bitte in 20 Sekunden nochmal drücken.")
            else:
                st.error(f"Fehler {response.status_code}: {response.text}")
