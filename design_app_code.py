import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Regenbogen Fundkiste", 
    page_icon="🌈", 
    layout="centered"
)

# --- REGENBOGEN & GLAS-DESIGN (CSS) ---
st.markdown("""
    <style>
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
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    /* Hüpfendes Schul-Emoji */
    .bounce {
        font-size: 100px;
        text-align: center;
        animation: bounce 2s infinite;
        margin-bottom: 10px;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-20px);}
        60% {transform: translateY(-10px);}
    }

    .welcome-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        background: -webkit-linear-gradient(#000, #444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Stylischer Button */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: #000;
        color: white;
        padding: 15px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.4s;
    }
    
    .stButton>button:hover {
        background: #444;
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar anpassen */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header Bereich
st.markdown('<div class="bounce">🏫</div>', unsafe_allow_html=True)
st.markdown('<p class="welcome-title">KI FUNDKISTE</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #333;'>Lade ein Bild hoch und erlebe Magie! ✨</p>", unsafe_allow_html=True)

# 3. Das Haupt-Layout in einer Glas-Karte
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Upload Bereich
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Spalten-Layout innerhalb der Karte
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.image(image, use_container_width=True)
            
        with col2:
            st.write("### 🤖 KI Analyse")
            if st.button("WAS IST DAS?"):
                # Simulation / Check System
                try:
                    from transformers import pipeline
                    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                    
                    with st.spinner("Die KI scannt..."):
                        results = classifier(image)
                        top = results[0]
                        st.balloons() # Party-Effekt bei Erfolg!
                        st.success(f"Ich erkenne: {top['label']}")
                        
                        for res in results:
                            st.write(f"{res['label']}")
                            st.progress(res['score'])
                except:
                    st.error("System wird noch geladen... bitte nochmal drücken!")
    else:
        st.info("Klicke oben, um ein Foto deiner Fundkiste hinzuzufügen! 📸")

    st.markdown('</div>', unsafe_allow_html=True)

# 4. Sidebar für Zusatz-Infos
with st.sidebar:
    st.title("🌈 Infos")
    st.write("Diese App wurde speziell für ein besonders buntes Erlebnis designt!")
    st.divider()
    st.markdown("Made with ❤️ & Streamlit")
