import streamlit as st
from PIL import Image
from model_loader import predict_vehicle

st.set_page_config(page_title="Vehicle Make & Model Recognition", layout="centered")
st.title("🚘 Vehicle Make & Model Recognition")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = predict_vehicle(image)

    st.markdown("### 🔍 Detection Results")
    if not results:
        st.warning("No vehicles detected.")
    else:
        for res in results:
            st.write(f"**Label:** {res['label']}  |  **Confidence:** {res['score']}")

