import streamlit as st
import cv2
import numpy as np
from histogram_equalization.global_equalization import global_equalization
from histogram_equalization.local_equalization import local_equalization
from histogram_equalization.clahe_equalization import clahe_equalization
from bit_plane_slicing.bit_plane import bit_plane_slicing

st.set_page_config(page_title="Image Processing WebApp", layout="wide")

# Scoped CSS injection
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    h1, h2, h3, h4 {
        color: #006699; /* headings styled */
    }
    /* Do NOT override selectbox or input styles */
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌐 Digital Image Processing WebApp")
st.write("Upload an image and perform Global Histogram Equalization (GHE), Local Histogram Equalization (LHE), CLAHE, or Bit Plane Slicing.")

uploaded_file = st.file_uploader("Upload a colourful image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # keep colour
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # convert to RGB for Streamlit
    st.image(img_rgb, caption="Original Image", width=700)

    st.subheader("Choose Operation")
    option = st.selectbox(
        "Select a method",
        ["Global Histogram Equalization (GHE)", 
         "Local Histogram Equalization (LHE)", 
         "CLAHE", 
         "Bit Plane Slicing"]
    )

    if option == "Global Histogram Equalization (GHE)":
        result = global_equalization(img)
        st.image(result, caption="GHE Result", width=700)

    elif option == "Local Histogram Equalization (LHE)":
        result = local_equalization(img)
        st.image(result, caption="LHE Result", width=700)

    elif option == "CLAHE":
        result = clahe_equalization(img)
        st.image(result, caption="CLAHE Result", width=700)

    elif option == "Bit Plane Slicing":
        fig = bit_plane_slicing(img)
        st.pyplot(fig)
