import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Skyline Fundkiste", 
    page_icon="🏙️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CLEAN SKYLINE DESIGN (CSS) ---
st.markdown("""
    <style>
    /* Sidebar komplett ausblenden */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        display: none;
    }
    
    /* Hintergrund: Das originale Skyline-Bild ohne Farbveränderung */
    .stApp {
        background: url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }

    /* Glas-Karte für den Inhalt (etwas weißer für bessere Lesbarkeit auf dem Foto) */
    .main-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 30px;
        margin-top: 50px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.5);
    }

    /* FLUGZEUG BANNER: Das eingeflogene Ergebnis */
    .fly-in-result {
        background: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 15px;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        text-transform: uppercase;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        
        /* Animation: Kommt von rechts reingeflogen */
        animation: flyIn 1s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    @keyframes flyIn {
        0% { transform: translateX(100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
        padding: 10px;
        border: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>🏙️ KI-Tower Analyse</h1>", unsafe_allow_html=True)

# 3. Haupt-Inhalt
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Bild zur Erkennung hochladen", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("OBJEKT SCANNER STARTEN"):
            try:
                from transformers import pipeline
                classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                
                with st.spinner("Berechne Daten..."):
                    results = classifier(image)
                    # Nur das erste Ergebnis (Label) nehmen
                    best_match = results[0]['label']

                    # Das fliegende Banner ohne Prozentzahlen
                    st.markdown(f"""
                        <div class="fly-in-result">
                            ✈️ GEFUNDEN: {best_match.upper()}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
            except:
                st.error("System lädt noch...")
    else:
        st.info("Bereit für den Upload. Das Skyline-System wartet auf Daten.")

    st.markdown('</div>', unsafe_allow_html=True)

# 4. Kleiner Footer
st.markdown("<p style='text-align: center; color: white; margin-top: 30px; font-weight: bold;'>Guten Flug über die Skyline! 🌤️</p>", unsafe_allow_html=True)
