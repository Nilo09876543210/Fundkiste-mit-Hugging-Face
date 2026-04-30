import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten für High-End Look
st.set_page_config(
    page_title="Vision Pro | AI", 
    page_icon="🏙️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL DARK DESIGN & AERODYNAMIC ANIMATION ---
st.markdown("""
    <style>
    /* Hintergrund: Tiefes Anthrazit zu Dunkelblau */
    .stApp {
        background: radial-gradient(circle at top, #1a1a2e 0%, #0f0f1a 100%);
        color: #e0e0e0;
    }

    /* Sidebar und Header-Elemente komplett ausblenden */
    [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
    
    /* Haupt-Container: Minimalistischer Glas-Look */
    .main-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        margin-top: 50px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    /* Professional Headline */
    .brand-title {
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-size: 14px;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 40px;
    }

    /* FLUGZEUG BANNER: Edel & Dezent */
    .pro-flight-banner {
        background: linear-gradient(90deg, #d4af37, #f1c40f); /* Gold-Look */
        color: #000;
        padding: 15px 30px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 28px;
        text-align: center;
        letter-spacing: 1px;
        position: relative;
        margin-top: 30px;
        box-shadow: 0 10px 20px rgba(212, 175, 55, 0.2);
        
        /* Die Einflug-Animation */
        animation: aerodynamicFlyIn 1.2s cubic-bezier(0.23, 1, 0.32, 1) forwards;
    }

    @keyframes aerodynamicFlyIn {
        0% { transform: translateX(-100%) skewX(20deg); opacity: 0; }
        100% { transform: translateX(0) skewX(0deg); opacity: 1; }
    }

    /* Button Styling: Dezent & Modern */
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 4px;
        padding: 10px;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: 0.4s all;
    }
    
    .stButton>button:hover {
        background-color: #d4af37;
        color: #000;
        border: 1px solid #d4af37;
    }

    /* Skyline Footer: Fixiert, Dezent, Hochwertig */
    .skyline-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 25vh;
        background: url('https://images.unsplash.com/photo-1449824913935-59a10b8d2000?q=80&w=2000') no-repeat bottom;
        background-size: cover;
        opacity: 0.3;
        mask-image: linear-gradient(transparent, black);
        z-index: -1;
    }

    /* Drag & Drop Box anpassen */
    .stFileUploader {
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 10px;
    }
    </style>
    
    <div class="skyline-footer"></div>
    """, unsafe_allow_html=True)

# 2. Header
st.markdown('<p class="brand-title">AERO VISION SYSTEM v3.0</p>', unsafe_allow_html=True)

# 3. Main Interface
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("IMAGE UPLOAD", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Hochwertige Bilddarstellung
        st.image(image, use_container_width=True)
        
        if st.button("RUN ANALYSIS"):
            try:
                from transformers import pipeline
                # Wir laden das Modell im Hintergrund
                classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                
                with st.spinner("PROCESSING..."):
                    results = classifier(image)
                    # Nur das erste Ergebnis ohne Score
                    top_label = results[0]['label']

                    # Der "Flug"-Effekt für das Ergebnis
                    st.markdown(f"""
                        <div class="pro-flight-banner">
                            {top_label.upper()}
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error("System initializing...")
    else:
        st.markdown("<p style='text-align: center; color: #555; font-size: 12px;'>WAITING FOR INPUT DATA</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 4. Minimaler Footer
st.markdown("<p style='text-align: center; color: #333; font-size: 10px; margin-top: 50px; letter-spacing: 2px;'>SYSTEM STATUS: OPERATIONAL</p>", unsafe_allow_html=True)
