import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. PAGE CONFIGURATION & DEBUGGING
# ==========================================
st.set_page_config(page_title="Diabetic Retinopathy AI", page_icon="👁️", layout="wide")

st.title("👁️ Diabetic Retinopathy Screening AI")
st.markdown("Upload a retinal fundus image below. The AI will apply medical CLAHE preprocessing and predict the severity of Diabetic Retinopathy.")

st.sidebar.info(f"TF Version Running: {tf.__version__}")

# ==========================================
# 2. LOAD THE AI MODEL
# ==========================================
@st.cache_resource
def load_ai_brain():
    # Back to the trusty .h5 format
    return load_model('final_clahe_model_V4.h5', compile=False)

try:
    model = load_ai_brain()
    st.sidebar.success("✅ AI Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model. Ensure 'final_clahe_model_V4.h5' is in the same folder as app.py. Error: {e}")
    st.stop()

# ==========================================
# 3. IMAGE VALIDATOR & PREPROCESSING
# ==========================================
def validate_retinal_image(img_array):
    """
    Heuristic check to determine if an image is a fundus image.
    Biologically, retinal images are highly vascular (red/orange) and absorb blue light.
    """
    # Calculate the mean intensity of the Red and Blue channels
    r_mean = np.mean(img_array[:, :, 0])
    b_mean = np.mean(img_array[:, :, 2])
    
    # In a real fundus image, the Red channel is significantly stronger than the Blue channel.
    # If Red is not at least 30% stronger than Blue, it's likely not a retinal image.
    if r_mean > (b_mean * 1.3): 
        return True
    return False

def preprocess_for_ai(img_array):
    # Extract the Green Channel
    g = img_array[:, :, 1]
    g_resized = cv2.resize(g, (300, 300))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_clahe = clahe.apply(g_resized)
    
    # Merge channels
    processed_img = cv2.merge([g_clahe, g_clahe, g_clahe])
    
    img_tensor = np.expand_dims(processed_img, axis=0)
    img_tensor = preprocess_input(img_tensor.astype('float32'))
    
    return img_tensor, processed_img

# ==========================================
# 4. USER INTERFACE & PREDICTION
# ==========================================
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Read image once at the top level
    pil_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(pil_image)
    
    # --- VALIDATION STEP ---
    if not validate_retinal_image(img_array):
        st.error("🚨 Error: The uploaded file does not appear to be a valid retinal fundus image. Please re-upload a correct retina image.")
    else:
        # Proceed with normal processing if validation passes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_container_width=True) # <--- NEW
            
        with st.spinner("Applying CLAHE Preprocessing & Analyzing..."):
            tensor, display_processed = preprocess_for_ai(img_array)
            
            with col2:
                st.subheader("AI Vision (CLAHE Green Channel)")
                st.image(display_processed, use_container_width=True, clamp=True) # <--- NEW
                
            predictions = model.predict(tensor)[0]
            predicted_class_index = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100
            
            labels = ['Normal (Grade 0)', 'Mild (Grade 1)', 'Moderate (Grade 2)', 'Severe (Grade 3)']
            result = labels[predicted_class_index]
            
        # ==========================================
        # 5. DISPLAY RESULTS
        # ==========================================
        st.markdown("---")
        st.subheader("📋 Diagnosis Report")
        
        if predicted_class_index == 0:
            st.success(f"**Prediction:** {result} (Confidence: {confidence:.2f}%)")
            st.info("Recommendation: No immediate action required. Routine annual checkup advised.")
        elif predicted_class_index == 1:
            st.warning(f"**Prediction:** {result} (Confidence: {confidence:.2f}%)")
            st.warning("Recommendation: Early signs detected. Schedule a follow-up with an ophthalmologist.")
        elif predicted_class_index == 2:
            st.error(f"**Prediction:** {result} (Confidence: {confidence:.2f}%)")
            st.error("Recommendation: Moderate damage detected. Prompt medical referral required.")
        else:
            st.error(f"**🚨 Prediction:** {result} (Confidence: {confidence:.2f}%)")
            st.error("🚨 Critical Recommendation: Severe DR detected. Immediate intervention required to prevent vision loss.")

        with st.expander("Show Detailed AI Probabilities"):
            for i, label in enumerate(labels):
                st.write(f"{label}: {predictions[i]*100:.2f}%")