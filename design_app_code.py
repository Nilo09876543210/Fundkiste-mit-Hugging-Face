import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Skyline Vision", 
    page_icon="🏙️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- TRANSPARENT FLOATING DESIGN (CSS) ---
st.markdown("""
    <style>
    /* UI-Elemente von Streamlit verstecken */
    [data-testid="stSidebar"], [data-testid="stHeader"], #tabs-bui3-tabpanel-0 {
        display: none;
    }
    
    /* Hintergrund: Die originale Skyline vollflächig */
    .stApp {
        background: url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }

    /* Schwebender Container */
    .main-card {
        background: rgba(255, 255, 255, 0.0);
        border-radius: 20px;
        padding: 10px;
        margin-top: 30px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.9);
        color: white;
    }

    /* Bild-Styling */
    [data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 30px 60px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* DAS EINE ERGEBNIS-BANNER */
    .single-result-banner {
        background: #ff4b4b;
        color: white;
        padding: 25px;
        border-radius: 60px;
        font-size: 38px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        margin-top: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        text-shadow: none;
        
        /* Einflug-Animation */
        animation: flyInHard 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    }

    @keyframes flyInHard {
        0% { transform: translateX(150%) scale(0.5); opacity: 0; }
        100% { transform: translateX(0) scale(1); opacity: 1; }
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        border-radius: 50px;
        padding: 15px;
        border: 2px solid white;
        font-weight: bold;
        backdrop-filter: blur(8px);
        font-size: 18px;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        box-shadow: 0 0 20px rgba(255,255,255,0.4);
    }

    /* Uploader-Bereich */
    .stFileUploader section {
        background-color: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.3) !important;
        border-radius: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Titel
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 4px 4px 15px rgba(0,0,0,1); letter-spacing: 3px;'>SKYLINE SCANNER</h1>", unsafe_allow_html=True)

# 3. Hauptbereich
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Analyse-Button
        if st.button("OBJEKT ERKENNEN"):
            try:
                from transformers import pipeline
                # Modell laden
                classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                
                with st.spinner("Processing..."):
                    results = classifier(image)
                    
                    # ENTSCHEIDEND: Nur das allererste Ergebnis nehmen
                    best_match = results[0]['label']
                    
                    # Falls "Jersey" oder ähnliches kommt, könnte man es hier noch filtern,
                    # aber die KI nimmt automatisch das wahrscheinlichste Objekt.
                    
                    st.markdown(f"""
                        <div class="single-result-banner">
                            🚀 {best_match.upper()}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
            except:
                st.error("AI Engine lädt noch... bitte kurz warten.")
    else:
        st.markdown("<p style='text-align: center; font-size: 1.2em; opacity: 0.8;'>Warte auf Bild-Input für den Skyline-Scan...</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
