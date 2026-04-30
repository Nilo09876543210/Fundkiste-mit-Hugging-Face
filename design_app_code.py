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
    /* Sidebar und Header weg */
    [data-testid="stSidebar"], [data-testid="stHeader"] {
        display: none;
    }
    
    /* Hintergrund: Die originale Skyline vollflächig */
    .stApp {
        background: url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000') no-repeat center center fixed;
        background-size: cover;
    }

    /* Der "Schwebe-Effekt": Das weisse Quadrat ist jetzt durchsichtig */
    .main-card {
        background: rgba(255, 255, 255, 0.0); /* Komplett durchsichtig */
        border-radius: 20px;
        padding: 20px;
        margin-top: 50px;
        /* Ein starker Text-Schatten hilft, damit alles auf dem Foto lesbar bleibt */
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        color: white;
    }

    /* Das hochgeladene Bild bekommt einen weichen Rand, um "echter" zu wirken */
    [data-testid="stImage"] img {
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }

    /* FLUGZEUG BANNER: Kräftiges Rot, damit es vor der Skyline knallt */
    .fly-in-result {
        background: #ff4b4b;
        color: white;
        padding: 25px;
        border-radius: 50px; /* Abgerundete Kapsel-Form */
        font-size: 35px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        margin-top: 30px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
        text-shadow: none; /* Kein Schatten auf dem roten Banner für Clean-Look */
        
        animation: flyIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    @keyframes flyIn {
        0% { transform: translateX(120%) rotate(10deg); opacity: 0; }
        100% { transform: translateX(0) rotate(0deg); opacity: 1; }
    }

    /* Button Styling: Glas-Optik */
    .stButton>button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 50px;
        padding: 15px;
        border: 2px solid white;
        font-weight: bold;
        backdrop-filter: blur(5px);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
    }

    /* Datei-Uploader Textfarbe anpassen */
    .stMarkdown, p, label {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 3px 3px 10px rgba(0,0,0,0.7);'>SKYLINE VISION</h1>", unsafe_allow_html=True)

# 3. Haupt-Inhalt (Schwebend)
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("FOTO HOCHLADEN", type=["jpg", "png", "jpeg"], label_visibility="hidden")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("OBJEKT IDENTIFIZIEREN"):
            try:
                from transformers import pipeline
                classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                
                with st.spinner("Scanning..."):
                    results = classifier(image)
                    best_match = results[0]['label']

                    # Nur das Label, eingeflogen
                    st.markdown(f"""
                        <div class="fly-in-result">
                            🚀 {best_match.upper()}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
            except:
                st.error("System-Check läuft noch...")
    else:
        st.markdown("<p style='text-align: center;'>Zieh ein Foto hierher, um den Scan zu starten.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
