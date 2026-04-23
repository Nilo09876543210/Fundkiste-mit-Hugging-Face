import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(page_title="Fundkiste Analyse", page_icon="🔍")
st.title("🔍 Fundkiste: Bild-Analyse")
st.write("Lade ein Bild hoch, und die KI sagt dir in einfachem Text, was sie erkennt.")

# 2. System-Check (Prüfen, ob alles installiert ist)
try:
    from transformers import pipeline
    system_bereit = True
except ImportError:
    system_bereit = False

if not system_bereit:
    st.warning("⏳ Der Server installiert noch die KI-Bausteine... Bitte hab 2-3 Minuten Geduld.")
    st.info("Wenn diese Meldung nicht verschwindet, klicke in Streamlit auf 'Reboot App'.")
    st.stop()

# 3. KI-Modell laden
@st.cache_resource
def load_model():
    # Nutzt ein zuverlässiges Modell von Google
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner('KI wird vorbereitet...'):
    classifier = load_model()

# 4. Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild aus (JPG, PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Bild öffnen
    image = Image.open(uploaded_file)
    
    # NumPy-Teil: Bild-Informationen auslesen
    img_array = np.array(image)
    st.write(f"**Bild-Info:** Das Bild hat eine Auflösung von {img_array.shape[1]}x{img_array.shape[0]} Pixeln.")
    
    # Bild anzeigen (mit dem neuen Breiten-Parameter)
    st.image(image, caption="Dein hochgeladenes Bild", width='stretch')
    
    # 5. Analyse-Button
    if st.button("Was ist auf dem Bild zu sehen?"):
        with st.spinner("KI analysiert..."):
            results = classifier(image)
            
            st.subheader("Ergebnis der Analyse:")
            
            # Ergebnisse als schönen Text ausgeben
            for res in results:
                name = res['label']
                # Umwandlung in Prozent
                prozent = round(res['score'] * 100, 1)
                
                # Schöne Textausgabe statt Code-Block
                st.markdown(f"### 📦 {name}")
                st.write(f"Die KI ist sich zu **{prozent}%** sicher, dass dies ein(e) **{name}** ist.")
                st.divider() # Trennlinie für bessere Lesbarkeit

else:
    st.info("Bitte lade oben ein Bild hoch, um die Analyse zu starten.")
