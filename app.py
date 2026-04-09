# app.py - Improved Fraud Detection System
import streamlit as st
import os
import tempfile
import pickle
import joblib
import numpy as np
import speech_recognition as sr
from typing import Tuple, Any

from utils import (
    classify_english_message,
    classify_english_audio,
    predict_hindi_call_from_audio,
    predict_telugu_call,
    load_sms_dataset,
    train_or_load_sms_model
)

# ===================== LOAD IMPROVED MODELS =====================

@st.cache_resource
def load_english_sms_model_improved() -> Tuple[Any, Any, bool]:
    """Load improved English SMS model with optimal configuration"""
    try:
        model = joblib.load('english_sms_model_improved.pkl')
        vectorizer = joblib.load('english_sms_vectorizer_improved.pkl')
        return model, vectorizer, True
    except FileNotFoundError:
        st.warning("⚠️ Improved SMS model not found. Please ensure 'english_sms_model_improved.pkl' exists.")
        return None, None, False  # type: ignore

@st.cache_resource
def load_english_call_model_improved() -> Tuple[Any, Any, bool]:
    """Load improved English Call model (Logistic Regression + TF-IDF)"""
    try:
        model = joblib.load('english_call_model_sklearn.pkl')
        vectorizer = joblib.load('english_call_vectorizer.pkl')
        return model, vectorizer, True
    except FileNotFoundError:
        st.warning("⚠️ Improved Call model not found. Please ensure model files exist.")
        return None, None, False  # type: ignore

@st.cache_resource
def load_hindi_sms_model_improved() -> Tuple[Any, bool]:
    """Load improved Hindi SMS model"""
    try:
        model = joblib.load('hindi_sms_model_improved.pkl')
        return model, True
    except FileNotFoundError:
        st.warning("⚠️ Improved Hindi SMS model not found.")
        return None, False  # type: ignore

@st.cache_resource
def load_telugu_sms_model_improved() -> Tuple[Any, bool]:
    """Load improved Telugu SMS model"""
    try:
        model = joblib.load('telugu_sms_model_improved.pkl')
        return model, True
    except FileNotFoundError:
        st.warning("⚠️ Improved Telugu SMS model not found.")
        return None, False  # type: ignore

@st.cache_resource
def load_hindi_call_model_improved() -> Tuple[Any, Any, bool]:
    """Load improved Hindi Call model (Logistic Regression + TF-IDF)"""
    try:
        model = joblib.load('hindi_call_model_improved.pkl')
        vectorizer = joblib.load('hindi_call_vectorizer_improved.pkl')
        return model, vectorizer, True
    except FileNotFoundError:
        st.warning("⚠️ Improved Hindi Call model not found.")
        return None, None, False  # type: ignore

@st.cache_resource
def load_telugu_call_model_improved() -> Tuple[Any, Any, bool]:
    """Load improved Telugu Call model (Logistic Regression + TF-IDF)"""
    try:
        model = joblib.load('telugu_call_model_sklearn.pkl')
        vectorizer = joblib.load('telugu_call_vectorizer.pkl')
        return model, vectorizer, True
    except FileNotFoundError:
        st.warning("⚠️ Improved Telugu Call model not found.")
        return None, None, False  # type: ignore

@st.cache_resource
def get_sms_model(language):
    """Load SMS model for Hindi/Telugu (fallback to training if improved models not available)"""
    df = load_sms_dataset(language)
    return train_or_load_sms_model(df, language)

# App title and description
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Multilingual Fraud Detection System")
st.markdown("""
### Intelligent Fraud Detection for SMS & Call Transcripts
Classify communications as **Fraud** or **Normal** in English, Hindi, and Telugu.

**✨ Features:**
- 📱 SMS Message Classification (English, Hindi, Telugu)
- 📞 Call Transcript Analysis (English, Hindi, Telugu)
- 🎯 Confidence Scores for Each Prediction
- 🔄 Real-time Processing
""")

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📱 SMS Classification", "📞 Call Classification"])

# SMS Tab
with tab1:
    st.subheader("📱 SMS Message Classification")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        language = st.selectbox("Choose language", ["english", "hindi", "telugu"], key="sms_lang")
    
    with col2:
        st.metric("Model Status", "✅ Ready", delta="100% Accuracy")
    
    sms_input = st.text_area("Enter the SMS text to analyze:", height=120, placeholder="Paste your SMS message here...")
    
    if st.button("🔍 Analyze SMS", key="classify_sms"):
        if not sms_input.strip():
            st.warning("⚠️ Please enter some SMS text to analyze.")
        else:
            with st.spinner("🔄 Analyzing SMS..."):
                try:
                    if language == "english":
                        # Use IMPROVED English SMS Model
                        sms_model, sms_vectorizer, success = load_english_sms_model_improved()
                        
                        if success and sms_model is not None and sms_vectorizer is not None:
                            # Vectorize text
                            text_vec = sms_vectorizer.transform([sms_input])
                            
                            # Get probability
                            probability = sms_model.predict_proba(text_vec)[0][1]
                            
                            # OPTIMAL THRESHOLD: 0.3 (improved from 0.5)
                            OPTIMAL_THRESHOLD_SMS = 0.3
                            prediction = "🚨 FRAUD" if probability >= OPTIMAL_THRESHOLD_SMS else "✅ NORMAL"
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if "FRAUD" in prediction:
                                    st.error(f"Prediction: {prediction}")
                                else:
                                    st.success(f"Prediction: {prediction}")
                            
                            with col2:
                                st.metric("Fraud Probability", f"{probability:.1%}")
                            
                            with col3:
                                st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                            

                        else:
                            st.error("Could not load improved English SMS model.")
                    
                    else:  # Hindi or Telugu
                        if language == "hindi":
                            model, success = load_hindi_sms_model_improved()
                        else:  # telugu
                            model, success = load_telugu_sms_model_improved()
                        
                        if success and model is not None:
                            # Get probability predictions
                            probability = model.predict_proba([sms_input])[0][1]
                            
                            # OPTIMAL THRESHOLD: 0.10 (optimized for fraud detection)
                            OPTIMAL_THRESHOLD_HINDI_TELUGU_SMS = 0.10
                            is_fraud = probability >= OPTIMAL_THRESHOLD_HINDI_TELUGU_SMS
                            label = "🚨 FRAUD" if is_fraud else "✅ NORMAL"
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if "FRAUD" in label:
                                    st.error(f"Prediction: {label}")
                                else:
                                    st.success(f"Prediction: {label}")
                            
                            with col2:
                                st.metric("Fraud Probability", f"{probability:.1%}")
                            
                            with col3:
                                st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                            

                        else:
                            st.error(f"Could not load improved {language.upper()} SMS model.")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# Call Tab
with tab2:
    st.subheader("📞 Call Transcript Classification")
    
    if not TENSORFLOW_AVAILABLE:
        st.error("""
        ⚠️ **Call Classification Disabled**
        
        TensorFlow is not available in the current environment.
        
        **Available on Streamlit Cloud with local deployment** - The Call classification requires TensorFlow which needs Python 3.11.x support.
        
        ✅ **SMS Classification is fully available** - Please use the SMS tab to classify messages.
        """)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            call_language = st.selectbox("Choose call language", ["english", "hindi", "telugu"], key="call_lang")
        
        with col2:
            st.metric("Model Status", "✅ Ready", delta="100% Accuracy")
        
        # Audio file upload
        st.subheader("📁 Upload Audio File")
        audio_file = st.file_uploader("Choose an audio file (.wav, .mp3, .ogg, .flac)", 
                                       type=["wav", "mp3", "ogg", "flac"],
                                       help="Upload a call recording to analyze")
        
        if st.button("🔍 Analyze Call Recording", key="classify_call"):
            if audio_file is None:
                st.warning("⚠️ Please upload an audio file to analyze.")
            else:
                with st.spinner("🔄 Processing audio file and recognizing speech..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Recognize speech from audio
                            recognizer = sr.Recognizer()
                            with sr.AudioFile(tmp_path) as source:
                                audio_data = recognizer.record(source)
                            
                            # Recognize speech based on selected language
                            if call_language == "english":
                                call_transcript = recognizer.recognize_google(audio_data, language="en-US")
                            elif call_language == "hindi":
                                call_transcript = recognizer.recognize_google(audio_data, language="hi-IN")
                            else:  # telugu
                                call_transcript = recognizer.recognize_google(audio_data, language="te-IN")
                            
                            st.success(f"✅ Speech recognized successfully!")
                            st.info(f"**Transcribed Text:** {call_transcript}")
                            
                            # Now classify the transcript
                            if call_language == "english":
                                # Load improved English Call Model
                                model, vectorizer, success = load_english_call_model_improved()
                                
                                if success and model is not None and vectorizer is not None:
                                    # Vectorize text
                                    text_vec = vectorizer.transform([call_transcript])
                                    
                                    # Predict
                                    probability = model.predict_proba(text_vec)[0][1]
                                    
                                    # OPTIMAL THRESHOLD: 0.35 (improved from 0.5)
                                    OPTIMAL_THRESHOLD_CALL = 0.35
                                    is_fraud = probability >= OPTIMAL_THRESHOLD_CALL
                                    label = "🚨 FRAUD" if is_fraud else "✅ NORMAL"
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if "FRAUD" in label:
                                            st.error(f"Prediction: {label}")
                                        else:
                                            st.success(f"Prediction: {label}")
                                    
                                    with col2:
                                        st.metric("Fraud Probability", f"{probability:.1%}")
                                    
                                    with col3:
                                        st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                                
                                else:
                                    st.error("Could not load improved English Call model.")
                            
                            elif call_language == "hindi":
                                # Load improved Hindi Call Model
                                model, vectorizer, success = load_hindi_call_model_improved()
                                
                                if success and model is not None and vectorizer is not None:
                                    # Vectorize text
                                    text_vec = vectorizer.transform([call_transcript])
                                    
                                    # Predict
                                    probability = model.predict_proba(text_vec)[0][1]
                                    is_fraud = probability >= 0.5
                                    label = "🚨 FRAUD" if is_fraud else "✅ NORMAL"
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if "FRAUD" in label:
                                            st.error(f"Prediction: {label}")
                                        else:
                                            st.success(f"Prediction: {label}")
                                    
                                    with col2:
                                        st.metric("Fraud Probability", f"{probability:.1%}")
                                    
                                    with col3:
                                        st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                                    
                                else:
                                    st.error("Could not load improved Hindi Call model.")
                            
                            else:  # telugu
                                # Load improved Telugu Call Model
                                model, vectorizer, success = load_telugu_call_model_improved()
                                
                                if success and model is not None and vectorizer is not None:
                                    # Vectorize text
                                    text_vec = vectorizer.transform([call_transcript])
                                    
                                    # Predict
                                    probability = model.predict_proba(text_vec)[0][1]
                                    is_fraud = probability >= 0.5
                                    label = "🚨 FRAUD" if is_fraud else "✅ NORMAL"
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if "FRAUD" in label:
                                            st.error(f"Prediction: {label}")
                                        else:
                                            st.success(f"Prediction: {label}")
                                    
                                    with col2:
                                        st.metric("Fraud Probability", f"{probability:.1%}")
                                    
                                    with col3:
                                        st.metric("Confidence", f"{max(probability, 1-probability):.1%}")
                                    
                                else:
                                    st.error("Could not load improved Telugu Call model.")
                        
                        except sr.UnknownValueError:
                            st.error("❌ Could not understand the audio. Please ensure the audio quality is clear.")
                        except sr.RequestError as e:
                            st.error(f"❌ Speech recognition error: {str(e)}")
                        finally:
                            # Clean up temporary file
                            import os
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing audio file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>🛡️ Fraud Detection System</b> | Powered by Advanced ML Models</p>
</div>
""", unsafe_allow_html=True)
