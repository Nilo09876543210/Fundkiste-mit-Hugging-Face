import streamlit as st
from PIL import Image
import numpy as np

# 1. Seite einrichten
st.set_page_config(
    page_title="Skyline KI-Flieger", 
    page_icon="✈️", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- REGENBOGEN, SKYLINE & FLUGZEUG-ANIMATION (CSS) ---
st.markdown("""
    <style>
    /* Sidebar weg */
    [data-testid="stSidebar"] { display: none; }
    
    /* Regenbogen Hintergrund */
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

    /* Glas-Karte */
    .main-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        z-index: 1;
        position: relative;
    }

    /* FLUGZEUG ANIMATION */
    .fly-in {
        position: relative;
        background: #fff;
        border: 3px solid #ff4b4b;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        
        /* Die Animation */
        animation: flyInRight 2s ease-out forwards;
    }

    @keyframes flyInRight {
        0% { transform: translateX(150%) scale(0.5); opacity: 0; }
        70% { transform: translateX(-10%) scale(1.1); opacity: 1; }
        100% { transform: translateX(0) scale(1); opacity: 1; }
    }

    /* Skyline Footer */
    .skyline-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        z-index: 0;
    }
    .skyline-img {
        width: 100%;
        height: auto;
        display: block;
        opacity: 0.6;
    }
    
    .stButton>button {
        background: #ff4b4b;
        color: white;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
    
    <div class="skyline-footer">
        <img src="https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?q=80&w=2000&auto=format&fit=crop" class="skyline-img">
    </div>
    """, unsafe_allow_html=True)

# 2. Header
st.markdown("<h1 style='text-align: center; color: white;'>✈️ KI Fundkiste Airline</h1>", unsafe_allow_html=True)

# 3. Haupt-Box
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Bild für den Tower hochladen...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        if st.button("FLUG STARTEN & ANALYSIEREN"):
            try:
                from transformers import pipeline
                classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
                
                with st.spinner("Tower berechnet Landung..."):
                    results = classifier(image)
                    top_result = results[0]
                    label = top_result['label']
                    prob = round(top_result['score'] * 100, 1)

                    # DAS FLIEGENDE ERGEBNIS
                    st.markdown(f"""
                        <div class="fly-in">
                            🛩️ EINGETROFFEN:<br>
                            <span style="font-size: 40px;">{label.upper()}</span><br>
                            Sicherheit: {prob}%
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    
                    # Details (statisch darunter)
                    with st.expander("Weitere Details sehen"):
                        for res in results:
                            st.write(f"{res['label']}: {round(res['score']*100, 1)}%")
                            st.progress(res['score'])
            except:
                st.error("Treibstoff wird noch geladen... versuch es gleich nochmal!")
    else:
        st.info("Bitte lade ein Bild hoch, um das Flugzeug zu starten! 🛫")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: white; margin-top: 20px;'>Guten Flug! 🌤️</p>", unsafe_allow_html=True)
