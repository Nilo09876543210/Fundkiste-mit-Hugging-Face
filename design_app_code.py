import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Fundkiste KI", 
    page_icon="🎨", 
    layout="centered"
)

# --- BUNTE FARBEN & DESIGN (CSS) ---
st.markdown("""
    <style>
    /* Hintergrund der gesamten App */
    .stApp {
        background: linear-gradient(to bottom, #ffffff, #f0f7ff);
    }
    
    /* Die Sidebar farbig machen */
    [data-testid="stSidebar"] {
        background-color: #ffefd5;
    }

    /* Willkommens-Text Styling */
    .welcome-text {
        text-align: center;
        color: #ff4b4b;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    
    .sub-welcome {
        text-align: center;
        color: #555555;
        font-size: 20px;
        margin-bottom: 30px;
    }

    /* Der Analyse-Button in knalligem Orange/Rot */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        height: 3em;
        background: linear-gradient(to right, #ff4b2b, #ff416c);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Herzlich Willkommen Header
st.markdown('<p class="welcome-text">Herzlich Willkommen! 😊</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-welcome">Schön, dass du da bist. Lass uns gemeinsam schauen, was in deiner Fundkiste steckt!</p>', unsafe_allow_html=True)

# 3. Sidebar mit bunten Infos
with st.sidebar:
    st.markdown("## 🎈 Menü")
    st.write("Diese schlaue KI erkennt Objekte in Sekunden.")
    st.divider()
    st.success("Tipp: Nutze helle Bilder für beste Ergebnisse!")

# 4. KI-Modell laden (mit Error-Handling)
try:
    from transformers import pipeline
    system_bereit = True
except ImportError:
    system_bereit = False

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

if not system_bereit:
    st.warning("🌈 Die bunten Farben werden geladen... bitte kurz warten.")
    st.stop()

classifier = load_model()

# 5. Bild-Upload Bereich
st.markdown("### 📸 Lade hier dein Fundstück hoch:")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Layout: Bild und Analyse nebeneinander
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.image(image, caption="Dein Bild", use_container_width=True)
    
    with col2:
        st.write("### 🚀 Aktion")
        if st.button("Jetzt analysieren"):
            with st.spinner("🤖 Die KI scannt das Bild..."):
                results = classifier(image)
                
                st.write("---")
                st.write("### 💎 Ergebnis:")
                for res in results:
                    score = res['score']
                    label = res['label']
                    
                    # Bunte Balken je nach Sicherheit
                    st.write(f"**{label}** ({round(score*100, 1)}%)")
                    st.progress(score)
else:
    # Platzhalter-Bild oder Info
    st.info("Oben auf 'Browse files' klicken, um ein Foto zu wählen! ✨")
