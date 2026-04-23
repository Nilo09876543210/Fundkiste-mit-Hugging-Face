import streamlit as st
import requests
from PIL import Image

# Titel der App
st.title("Fundkiste Analyse")

# Datei-Uploader
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. FIX: Zeile war falsch eingerückt (IndentationError)
    image = Image.open(uploaded_file)
    
    # 2. FIX: 'width="stretch"' statt veraltetem 'use_container_width'
    st.image(image, caption='Hochgeladenes Bild', width='stretch')

    # Konfiguration für die Hugging Face API
    # Ersetze die URL durch deine tatsächliche Modell-URL
    API_URL = "https://api-inference.huggingface.co/models/DEIN_MODELL"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    if st.button("Analyse starten"):
        with st.spinner('Analysiere Bild...'):
            response = requests.post(API_URL, headers=headers, data=uploaded_file.getvalue())
            
            # 3. FIX: Prüfung des Status-Codes zur Vermeidung von JSONDecodeError
            if response.status_code == 200:
                try:
                    results = response.json()
                    st.success("Analyse abgeschlossen!")
                    st.write(results)
                except Exception as e:
                    st.error(f"Fehler beim Verarbeiten der Daten: {e}")
            elif response.status_code == 503:
                st.warning("Das KI-Modell startet gerade noch auf Hugging Face. Bitte warte ca. 20-30 Sekunden und klicke dann erneut auf den Button.")
            else:
                st.error(f"Fehler von der API: {response.status_code}")
                st.info("Hinweis: Überprüfe, ob dein API-Token (HF_TOKEN) in den Streamlit Secrets korrekt hinterlegt ist.")
