"""
Streamlit frontend for Voice Emotion Detection
User-friendly interface for emotion prediction
"""

import streamlit as st
import requests
import json
from io import BytesIO
import time

# Configuration
BACKEND_URL = "http://localhost:8000"  # Change to ngrok URL if using tunneling

# Page configuration
st.set_page_config(
    page_title="Voice Emotion Detection",
    page_icon="🎤",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-result {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_supported_emotions():
    """Get list of supported emotions from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/emotions", timeout=2)
        if response.status_code == 200:
            return response.json()["emotions"]
        return []
    except:
        return []


def predict_emotion(audio_file):
    """Send audio file to backend for prediction"""
    try:
        files = {"file": ("audio.wav", audio_file, "audio/wav")}
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None


def get_emotion_emoji(emotion):
    """Map emotion to emoji"""
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fearful": "😨",
        "disgust": "🤢",
        "surprised": "😲",
        "neutral": "😐",
        "calm": "😌"
    }
    return emoji_map.get(emotion.lower(), "🎭")


def get_emotion_color(emotion):
    """Map emotion to color"""
    color_map = {
        "happy": "#FFD700",
        "sad": "#4682B4",
        "angry": "#DC143C",
        "fearful": "#9370DB",
        "disgust": "#8B4513",
        "surprised": "#FF69B4",
        "neutral": "#A9A9A9",
        "calm": "#87CEEB"
    }
    return color_map.get(emotion.lower(), "#808080")


# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🎤 Voice Emotion Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an audio file to detect the emotion in the voice</p>',
                unsafe_allow_html=True)

    # Check backend status
    with st.sidebar:
        st.header("ℹ️ Information")

        backend_status = check_backend_health()
        if backend_status:
            st.success("✅ Backend connected")
            emotions = get_supported_emotions()
            if emotions:
                st.write("**Supported Emotions:**")
                for emotion in emotions:
                    emoji = get_emotion_emoji(emotion)
                    st.write(f"{emoji} {emotion.capitalize()}")
        else:
            st.error("❌ Backend not available")
            st.warning("Please start the FastAPI backend first:")
            st.code("cd backend && uvicorn main:app --reload")
            return

        st.markdown("---")
        st.write("**Instructions:**")
        st.write("1. Upload a .wav audio file")
        st.write("2. Wait for processing")
        st.write("3. View the predicted emotion")

        st.markdown("---")
        st.write("**Audio Requirements:**")
        st.write("• Format: WAV")
        st.write("• Duration: ~3-4 seconds")
        st.write("• Clear voice audio")

    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav"],
        help="Upload a WAV audio file containing voice"
    )

    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Audio player
        st.audio(uploaded_file, format="audio/wav")

        # Predict button
        if st.button("🔮 Predict Emotion", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing audio..."):
                # Reset file pointer
                uploaded_file.seek(0)

                # Get prediction
                result = predict_emotion(uploaded_file)

                if result:
                    if result["valid"]:
                        # Success - display results
                        emotion = result["emotion"]
                        confidence = result["confidence"]
                        emoji = get_emotion_emoji(emotion)
                        color = get_emotion_color(emotion)

                        # Display main result
                        st.markdown("---")
                        st.markdown(
                            f'<div class="emotion-result" style="background-color: {color}20; border: 3px solid {color};">'
                            f'{emoji} {emotion.upper()}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<p class="confidence-score">Confidence: {confidence:.1%}</p>',
                            unsafe_allow_html=True
                        )

                        # Confidence bar
                        st.progress(confidence)

                        # All probabilities
                        st.markdown("---")
                        st.subheader("📊 All Emotion Probabilities")

                        # Sort probabilities
                        sorted_probs = sorted(
                            result["all_probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )

                        for emo, prob in sorted_probs:
                            emoji_icon = get_emotion_emoji(emo)
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"{emoji_icon} **{emo.capitalize()}**")
                                st.progress(prob)
                            with col2:
                                st.write(f"{prob:.1%}")

                        # Success message
                        st.success("✅ Analysis complete!")

                    else:
                        # Invalid audio
                        st.error("❌ " + result.get("error", "Invalid audio file"))
                        st.info(
                            "💡 Please ensure:\n- The file is a valid WAV format\n- The audio contains clear voice\n- The audio is not too quiet")
                else:
                    st.error("Failed to get prediction from backend")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with FastAPI + Streamlit | Voice Emotion Recognition</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()