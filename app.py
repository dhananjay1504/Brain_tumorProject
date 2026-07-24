import json
import keras
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Brain Tumor Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Ultra-Vibrant Glassmorphism & Gradient CSS
# ---------------------------------------------------
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #31103f 100%);
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Vibrant Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #ff007a 0%, #7928ca 50%, #4338ca 100%);
        border-radius: 24px;
        padding: 35px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(255, 0, 122, 0.25);
    }
    
    /* Pill Badge */
    .pill-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 8px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }

    /* Frosted Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .glass-card h4 {
        color: #ffffff;
        font-weight: 800;
        font-size: 1.25rem;
        margin-bottom: 20px;
    }

    /* Metric Glass Cards */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    .metric-box-pink {
        flex: 1;
        min-width: 160px;
        background: linear-gradient(135deg, rgba(255, 0, 122, 0.2) 0%, rgba(121, 40, 202, 0.2) 100%);
        border: 1px solid rgba(255, 0, 122, 0.4);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-box-purple {
        flex: 1;
        min-width: 160px;
        background: linear-gradient(135deg, rgba(121, 40, 202, 0.2) 0%, rgba(67, 56, 202, 0.2) 100%);
        border: 1px solid rgba(121, 40, 202, 0.4);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 900;
        color: #ffffff;
        margin-top: 6px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    /* Vibrant Gradient Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff007a 0%, #7928ca 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 1.05rem;
        border-radius: 14px;
        padding: 14px;
        border: none;
        box-shadow: 0 10px 25px rgba(255, 0, 122, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #e0006c 0%, #6920b3 100%);
        box-shadow: 0 15px 30px rgba(255, 0, 122, 0.6);
        transform: translateY(-2px);
    }

    /* File Uploader Custom Styling */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 10px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Mobile Responsive Queries */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
        .hero-banner { padding: 20px; }
        .metric-box-pink, .metric-box-purple { flex: 100%; }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Config & Paths
# ---------------------------------------------------
CONFIG_PATH = "config.json"
WEIGHTS_PATH = "model.weights.h5"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = (224, 224)

# ---------------------------------------------------
# Model Loader
# ---------------------------------------------------
@st.cache_resource
def get_model():
    with open(CONFIG_PATH, "r") as f:
        model_json = f.read()
    model = keras.models.model_from_json(model_json)
    model.load_weights(WEIGHTS_PATH)
    return model

model = get_model()

# ---------------------------------------------------
# Sidebar Overview
# ---------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=70)
    st.markdown("<h3 style='color: #ff007a;'>System Details</h3>", unsafe_allow_html=True)
    st.write(
        "A Deep Learning decision support system for accurate multi-class brain tumor classification from MRI scans."
    )
    st.markdown("---")
    st.markdown("<h5 style='color: #a855f7;'>⚙️ Model Specs</h5>", unsafe_allow_html=True)
    st.markdown("""
    - **Architecture:** Convolutional Neural Network
    - **Classes:** Glioma, Meningioma, Pituitary, No Tumor
    - **Input Matrix:** 224x224 RGB
    - **Backend:** Keras 3 / TensorFlow
    """)
    st.markdown("---")
    st.caption("Developed for Research & Portfolio Presentation")

# ---------------------------------------------------
# Hero Banner (Similar to reference image style)
# ---------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <div class="pill-badge">🧠 Deep Learning AI Platform</div>
        <div class="hero-title">BRAIN TUMOR DIAGNOSIS</div>
        <div class="hero-subtitle">Real-time MRI Image Analysis & Multi-Class Classification System</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Main Content Grid
# ---------------------------------------------------
col_left, col_right = st.columns([1, 1.1], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>📸 Upload Brain MRI Scan</h4>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Scan", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Preview", use_container_width=True)
        
        # Preprocessing
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 RUN AI DIAGNOSTIC ANALYSIS")
    else:
        st.info("Please upload a brain MRI scan (JPG/PNG) to execute analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>📊 Diagnostic Output</h4>", unsafe_allow_html=True)
    
    if uploaded_file is not None and ('analyze_btn' in locals() and analyze_btn):
        with st.spinner("Extracting features through Convolutional Layers..."):
            prediction = model.predict(img_array)[0]
            top_class_idx = np.argmax(prediction)
            predicted_class = CLASSES[top_class_idx]
            confidence = prediction[top_class_idx] * 100

        # Glass Metric Cards
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-box-pink">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-val">{predicted_class}</div>
                </div>
                <div class="metric-box-purple">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-val">{confidence:.1f}%</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown("<p style='font-weight:700; color:rgba(255,255,255,0.8); margin-top:15px; margin-bottom:8px;'>Class Probability Breakdown</p>", unsafe_allow_html=True)

        # Plotly Glowing Horizontal Bar Chart
        colors = ['#ff007a' if i == top_class_idx else 'rgba(255, 255, 255, 0.2)' for i in range(len(CLASSES))]
        
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in prediction],
            y=CLASSES,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#ff007a', width=1.5)
            ),
            text=[f"{p * 100:.1f}%" for p in prediction],
            textposition='auto',
            textfont=dict(color='#ffffff', size=13)
        ))

        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis=dict(autorange="reversed", tickfont=dict(color='#ffffff')),
            xaxis=dict(range=[0, 100], tickfont=dict(color='#ffffff'), gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=10, r=10, t=10, b=30),
            height=260,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    elif uploaded_file is not None:
        st.write("👉 Click **RUN AI DIAGNOSTIC ANALYSIS** to generate prediction.")
    else:
        st.write("Awaiting image upload...")
    
    st.markdown('</div>', unsafe_allow_html=True)