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

# --- TOTAL TRANSPARENT IMAGE DESIGN (CSS) ---
st.markdown("""
    <style>
    /* Streamlit UI ausblenden */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        display: none;
    }
    
    /* Hintergrund: Die originale Skyline */
    .stApp {
        background: url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }

    /* Das Bild-Element selbst komplett freistellen */
    [data-testid="stImage"] {
        background-color: transparent !important; /* Weißer Hintergrund weg */
        border: none !important;
        display: flex;
        justify-content: center;
    }

    [data-testid="stImage"] img {
        background-color: transparent !important; /* Hintergrund des Bildes selbst weg */
        border-radius: 15px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.7); /* Schatten, damit es schwebt */
    }

    /* Uploader-Box fast unsichtbar machen */
    .stFileUploader section {
        background-color: rgba(255,255,255,0.02) !important;
        border: 1px dashed rgba(255,255,255,0.2) !important;
    }

    /* Das Ergebnis-Banner */
    .single-result-banner {
        background: #ff4b4b;
        color: white;
        padding: 20px 40px;
        border-radius: 50px;
        font-size: 35px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        margin-top: 30px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.5);
        animation: flyInHard 0.6s ease-out forwards;
    }

    @keyframes flyInHard {
        0% { transform: translateX(150%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* Button im Glas-Design */
    .stButton>button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid white;
        border-radius: 50px;
        padding: 12px;
        font-weight: bold;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Inhalt
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 10px #000;'>SKYLINE SCANNER</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    # Hier wird das Bild angezeigt - jetzt ohne weißen Kasten
    st.image(image, use_container_width=True)
    
    if st.button("OBJEKT ERKENNEN"):
        from transformers import pipeline
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        
        with st.spinner("Scanning..."):
            results = classifier(image)
            best_match = results[0]['label']
            
            st.markdown(f'<div class="single-result-banner">🚀 {best_match.upper()}</div>', unsafe_allow_html=True)
            st.balloons()
