import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Skyline Vision Pro", 
    page_icon="🌈", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- REGENBOGEN-RAHMEN & TRANSPARENT DESIGN (CSS) ---
st.markdown("""
    <style>
    /* UI ausblenden */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        display: none;
    }
    
    /* Hintergrund: Skyline */
    .stApp {
        background: url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }

    /* Bild-Container clean machen */
    [data-testid="stImage"] {
        background-color: transparent !important;
    }
    [data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }

    /* DER REGENBOGEN-RAHMEN EFFEKT */
    .rainbow-box {
        position: relative;
        padding: 5px; /* Dicke des Rahmens */
        background: linear-gradient(90deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff);
        background-size: 400% 400%;
        border-radius: 55px;
        animation: rainbow-animation 3s linear infinite, flyInHard 0.7s ease-out forwards;
        margin-top: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Innerer Teil des Banners (schwarz/dunkel für Kontrast) */
    .rainbow-content {
        background: rgba(0, 0, 0, 0.85);
        color: white;
        padding: 20px 40px;
        border-radius: 50px;
        font-size: 35px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        display: block;
    }

    /* Animation für das Wandern der Regenbogenfarben */
    @keyframes rainbow-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Einflug-Animation */
    @keyframes flyInHard {
        0% { transform: translateX(150%) rotate(5deg); opacity: 0; }
        100% { transform: translateX(0) rotate(0deg); opacity: 1; }
    }

    /* Glas-Button */
    .stButton>button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.15);
        color: white;
        border: 2px solid white;
        border-radius: 50px;
        padding: 15px;
        font-weight: bold;
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
    }

    /* Verstecke den Uploader-Text */
    .stMarkdown, p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. Titel
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 15px #000;'>SKYLINE SCANNER</h1>", unsafe_allow_html=True)

# 3. Logik
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    if st.button("OBJEKT ERKENNEN"):
        try:
            from transformers import pipeline
            classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            
            with st.spinner("Scanning..."):
                results = classifier(image)
                best_match = results[0]['label']
                
                # Das Banner mit dem animierten Regenbogen-Rahmen
                st.markdown(f"""
                    <div class="rainbow-box">
                        <div class="rainbow-content">
                            🚀 {best_match.upper()}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
        except:
            st.error("System-Start...")
else:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Bereit für den Scan über der Stadt.</p>", unsafe_allow_html=True)
