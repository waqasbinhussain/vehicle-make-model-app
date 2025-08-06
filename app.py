if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        result = predict_vehicle(image)

    st.markdown("### 🔍 Prediction")
    st.write(f"**Label:** {result['label']}  |  **Confidence:** {result['confidence']}")
