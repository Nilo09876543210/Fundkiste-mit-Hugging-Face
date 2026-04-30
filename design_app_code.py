import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten (Seitenleiste verstecken)
st.set_page_config(
    page_title="Regenbogen Fundkiste", 
    page_icon="🌈", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- REGENBOGEN, GLAS-EFFEKT & SKYLINE (CSS) ---
st.markdown("""
    <style>
    /* Seitenleiste komplett ausblenden */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Animierter Regenbogen-Hintergrund */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glas-Effekt Karte für den Inhalt */
    .main-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 30px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 100px; /* Platz für die Skyline unten */
        z-index: 1;
    }

    /* Hüpfendes Schul-Emoji */
    .bounce {
        font-size: 80px;
        text-align: center;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-20px);}
    }

    /* Skyline am unteren Rand fixieren */
    .skyline-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        z-index: 0;
        pointer-events: none; /* Man kann durch das Bild klicken */
    }

    .skyline-img {
        width: 100%;
        height: auto;
        display: block;
        opacity: 0.8; /* Leicht transparent für den Look */
    }

    .welcome-title {
        text-align: center;
        font-size: 40px;
        font-weight: 900;
        color: #111;
        margin-top: 0px;
    }

    /* Analyse Button */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background: #000;
        color: white;
        padding: 12px;
        font-size: 18px;
        border: none;
    }
    </style>
    
    <!-- Hier fügen wir das Skyline-Bild ein -->
    <div class="skyline-footer">
        <img src="https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000&auto=format&fit=crop" class="skyline-img">
    </div>
    """, unsafe_allow_html=True)

# 2. Header Bereich
st.markdown('<div class="bounce">🏫</div>', unsafe_allow_html=True)
st.markdown('<p class="welcome-title">KI FUNDKISTE</p>', unsafe_allow_html=True)

# 3. Haupt-Inhalt in der Glas-Karte
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Wähle ein Foto aus...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.image(image, use_container_width=True)
            
        with col2:
            st.write("### ✨ Scan")
            if st.button("ANALYSİEREN"):
                try:
                    from transformers import pipeline
                    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                    
                    with st.spinner("KI sucht..."):
                        results = classifier(image)
                        st.balloons()
                        st.success(f"Gefunden: {results[0]['label']}")
                        
                        for res in results:
                            st.write(f"**{res['label']}**")
                            st.progress(res['score'])
                except:
                    st.error("Warte kurz, die KI lädt noch...")
    else:
        st.info("Willkommen! Lade ein Bild hoch, um die Stadt der KI zu erkunden! 🌆")

    st.markdown('</div>', unsafe_allow_html=True)

# 4. Kleiner Footer-Text (über der Skyline)
st.markdown("<p style='text-align: center; color: white; position: relative; z-index: 2;'>Fundkiste v2.0 - Powered by AI</p>", unsafe_allow_html=True)
