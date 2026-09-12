import streamlit as st
import cv2
import numpy as np

# Image processing module imports (flat structure)
from global_equalization import global_equalization
from local_equalization import local_equalization
from clahe_equalization import clahe_equalization
from bit_plane import bit_plane_slicing

from discrete_wavelet_transform import process_haar_dwt
from edge_filters import process_edge_detection
from huffman import process_huffman_coding
from shannon_fano import process_shannon_fano_coding

st.set_page_config(page_title="Digital Image Processing WebApp", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    h1, h2, h3, h4 {
        color: #006699;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌐 Digital Image Processing WebApp")
st.write(
    "Upload an image from your device to perform Histogram Equalization, Bit Plane Slicing, "
    "Discrete Wavelet Transform, Edge Detection, Huffman Coding, or Shannon-Fano Coding."
)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Read image directly from GUI upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Input Image", width=700)

    st.subheader("Choose Operation")
    option = st.selectbox(
        "Select a processing method",
        [
            "Global Histogram Equalization (GHE)", 
            "Local Histogram Equalization (LHE)", 
            "CLAHE", 
            "Bit Plane Slicing",
            "Discrete Wavelet Transform (Haar DWT & IDWT)",
            "Edge Detection (1st & 2nd Order)",
            "Huffman Coding",
            "Shannon-Fano Coding"
        ]
    )

    if option == "Global Histogram Equalization (GHE)":
        result = global_equalization(img_bgr)
        st.image(result, caption="GHE Result", width=700)

    elif option == "Local Histogram Equalization (LHE)":
        result = local_equalization(img_bgr)
        st.image(result, caption="LHE Result", width=700)

    elif option == "CLAHE":
        result = clahe_equalization(img_bgr)
        st.image(result, caption="CLAHE Result", width=700)

    elif option == "Bit Plane Slicing":
        fig = bit_plane_slicing(img_bgr)
        st.pyplot(fig)

    elif option == "Discrete Wavelet Transform (Haar DWT & IDWT)":
        with st.spinner("Processing Discrete Wavelet Transform..."):
            dwt_res = process_haar_dwt(img_rgb)
            st.image(dwt_res["overview_fig"], caption="Discrete Wavelet Transform Overview", use_container_width=True)
            st.success(f"Perfect Reconstruction Verified | MSE: {dwt_res['mse']:.2e} | Max Diff: {dwt_res['max_err']:.2f}")

    elif option == "Edge Detection (1st & 2nd Order)":
        with st.spinner("Processing Edge Detection Filters..."):
            edge_res = process_edge_detection(img_rgb)
            st.image(edge_res["master_grid"], caption="Master Edge Detection Overview Grid", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(edge_res["grid_1st"], caption="1st Order Numerical Differences", use_container_width=True)
            with col2:
                st.image(edge_res["grid_2nd"], caption="2nd Order Operators (LoG & DoG Zero-Crossings)", use_container_width=True)

    elif option == "Huffman Coding":
        with st.spinner("Encoding and Decoding with Huffman Coding..."):
            huff_res = process_huffman_coding(img_rgb)
            st.image(huff_res["summary_card"], caption="Huffman Performance Metrics", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(huff_res["heatmap"], caption="Codeword Length Heatmap", use_container_width=True)
            with col2:
                st.image(huff_res["length_dist"], caption="Codeword Length Distribution", use_container_width=True)

            st.image(huff_res["decoded"], caption="Reconstructed Lossless Decoded Image", width=700)

    elif option == "Shannon-Fano Coding":
        with st.spinner("Encoding and Decoding with Shannon-Fano Coding..."):
            sf_res = process_shannon_fano_coding(img_rgb)
            st.image(sf_res["summary_card"], caption="Shannon-Fano Performance Metrics", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(sf_res["heatmap"], caption="Codeword Length Heatmap", use_container_width=True)
            with col2:
                st.image(sf_res["length_dist"], caption="Codeword Length Distribution", use_container_width=True)

            st.image(sf_res["decoded"], caption="Reconstructed Lossless Decoded Image", width=700)
else:
    st.info("Please upload an image file to begin processing.")