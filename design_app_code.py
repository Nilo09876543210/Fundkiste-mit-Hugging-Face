import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten (für Vollbild/weite Ansicht)
st.set_page_config(
    page_title="Fundkiste KI", 
    page_icon="🏫", 
    layout="wide" # Für mehr Platz
)

# --- BUNTE FARBEN & DESIGN (CSS) ---
st.markdown("""
    <style>
    /* Hintergrund der gesamten App: Hellblau */
    .stApp {
        background-color: #E0F7FA;
    }
    
    /* Die Sidebar farbig machen */
    [data-testid="stSidebar"] {
        background-color: #B2EBF2;
    }

    /* Großer Schul-Header Container */
    .school-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: -30px; /* Ein wenig nach oben ziehen */
        margin-bottom: 20px;
        color: #006064;
    }

    .school-emoji {
        font-size: 100px; /* Sehr großes Emoji */
        margin-bottom: 0px;
    }

    /* Willkommens-Text Styling */
    .welcome-text {
        text-align: center;
        color: #006064;
        font-size: 42px;
        font-weight: bold;
        margin-top: -10px;
    }
    
    .sub-welcome {
        text-align: center;
        color: #555555;
        font-size: 20px;
        margin-bottom: 30px;
    }

    /* Container für die Analyse-Sektion */
    .analysis-section {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Der Analyse-Button in knalligem Türkis */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        height: 3em;
        background: linear-gradient(to right, #00838F, #00ACC1);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 15px rgba(0, 128, 128, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 128, 128, 0.5);
    }
    
    /* Schriftfarbe für alle Standard-Texte in Weiß für besseren Kontrast auf Blau */
    p, stMarkdown, label {
        color: #333333;
    }
    
    </style>
    """, unsafe_allow_html=True)

# 2. Großer Schul-Header (Emoji und Willkommen)
st.markdown("""
    <div class="school-header">
        <p class="school-emoji">🏫</p>
        <p class="welcome-text">Willkommen in deiner KI-Fundkiste!</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('<p class="sub-welcome">Lade ein Foto hoch, und unsere clevere KI sagt dir, was es ist.</p>', unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.markdown("## 📖 Hinweise")
    st.write("Diese App hilft Schülern und Lehrern, Fundstücke schnell zu identifizieren.")
    st.divider()
    st.info("💡 Beste Ergebnisse bei gutem Licht und klarem Fokus auf das Objekt.")

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
    st.warning("🔄 Die KI wird hochgefahren... bitte 1-2 Minuten Geduld.")
    st.stop()

classifier = load_model()

# 5. Bild-Upload Bereich (zentriert)
st.markdown("### 📸 Schritt 1: Foto hochladen")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], help="Zieh dein Bild hierher oder klicke")

# Wenn ein Bild hochgeladen wurde, zeige die Analyse-Sektion
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Trennungslinie
    st.divider()
    
    # Ein Container, um die Analyse visuell zu umrahmen (Card-Design)
    with st.container():
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Schritt 2: Analyse")
        
        # Layout: Bild und Analyse nebeneinander
        col1, col2 = st.columns([1.2, 1], gap="large")
        
        with col1:
            st.image(image, caption="Dein Fundstück", use_container_width=True)
        
        with col2:
            st.write("#### 🚀 Aktion")
            st.write("Klicke auf den Button, um die Objekterkennung zu starten.")
            if st.button("Jetzt scannen"):
                with st.spinner("🤖 KI scann..."):
                    results = classifier(image)
                    
                    st.write("---")
                    st.write("### 🏆 Ergebnis:")
                    
                    # Das Hauptergebnis besonders hervorheben
                    top_result = results[0]
                    st.success(f"Hauptverdacht: **{top_result['label'].upper()}** ({round(top_result['score']*100, 1)}%)")
                    
                    # Alle Ergebnisse mit Fortschrittsbalken
                    for res in results:
                        score = res['score']
                        label = res['label']
                        st.write(f"**{label}**")
                        st.progress(score)
        
        st.markdown('</div>', unsafe_allow_html=True) # Ende der analysis-section
        
else:
    # Platzhalter, wenn kein Bild da ist
    st.divider()
    st.info("Startbereit! ✨ Lade oben ein Foto hoch, um zu beginnen.")
