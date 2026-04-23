# Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption='Dein Fundstück', use_container_width=True)
    
    # Bild für die API konvertieren
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    if st.button('Gegenstand erkennen 🔍'):
        with st.spinner('KI analysiert... (beim ersten Mal kann es kurz dauern)'):
            
            # Die Anfrage an Hugging Face
            response = requests.post(API_URL, headers=headers, data=byte_im)
            
            if response.status_code == 200:
                results = response.json()
                
                st.subheader("Ergebnis:")
                # Das beste Ergebnis anzeigen
                top_result = results[0]
                label = top_result['label']
                conf = round(top_result['score'] * 100, 1)
                
                st.success(f"Ich erkenne: **{label}**")
                st.progress(top_result['score'])
                st.write(f"Sicherheit: {conf}%")
                
            elif response.status_code == 503:
                st.warning("Das Modell schläft noch und wacht gerade auf. Bitte klicke in 20 Sekunden nochmal auf den Button.")
            elif response.status_code == 401:
                st.error("Fehler: Dein Hugging Face Token ist ungültig. Überprüfe deine Secrets.")
            else:
                st.error(f"Fehler: {response.status_code}")
                st.json(response.json()) # Zeigt die genaue Fehlermeldung der API
