import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import platform
import matplotlib.pyplot as plt
from collections import Counter
import threading
import av
import queue

# Enhanced WebRTC configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]}
)

# Initialize session state for emotion history if not present
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# Functions to handle emotion history using session state instead of JSON files
def append_to_emotion_history(data):
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    st.session_state.emotion_history.append(data)

def get_emotion_history():
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    return st.session_state.emotion_history

# Cross-platform TTS initialization
@st.cache_resource
def initialize_tts():
    # Detect platform and initialize appropriate TTS engine
    system = platform.system()

    if system == "Windows":
        # Windows-specific TTS
        try:
            import pyttsx3
            engine = pyttsx3.init()
            return {"type": "pyttsx3", "engine": engine}
        except Exception as e:
            st.warning(f"Windows TTS initialization failed: {e}")
            return {"type": "none", "engine": None}
    else:
        # For Linux/Mac, try gTTS (Google Text-to-Speech) if available
        try:
            from gtts import gTTS
            import tempfile
            import os
            from io import BytesIO

            def speak_gtts(text):
                try:
                    tts = gTTS(text=text, lang='en', slow=False)
                    # In cloud environments, we can't play audio directly
                    # but we can save the text that would be spoken
                    st.session_state.last_spoken = text
                except Exception as e:
                    st.warning(f"gTTS error: {e}")

            return {"type": "gtts", "engine": speak_gtts}
        except ImportError:
            # If gTTS is not available, try pyttsx3 as fallback
            try:
                import pyttsx3
                engine = pyttsx3.init()
                return {"type": "pyttsx3", "engine": engine}
            except Exception as e:
                # If all TTS options fail, return None
                st.warning(f"All TTS options failed: {e}")
                return {"type": "none", "engine": None}

tts_engine = initialize_tts()

# Load model and other resources with proper error handling
@st.cache_resource
def load_resources():
    try:
        # Ensure directories exist
        os.makedirs('Streamlit_FER_system-main', exist_ok=True)

        # Load the model
        model_path = 'Streamlit_FER_system-main/face_model.h5'
        if not os.path.exists(model_path):
            st.warning(f"Model file not found at {model_path}. Please ensure it's included in your repository.")
            return None, None

        model = load_model(model_path)

        # Load face cascade
        cascade_path = 'Streamlit_FER_system-main/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            st.warning(f"Cascade file not found at {cascade_path}. Please ensure it's included in your repository.")
            return model, None

        face_cascade = cv2.CascadeClassifier(cascade_path)
        return model, face_cascade
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None

# Load resources
model_best, face_cascade = load_resources()
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create a queue for TTS to avoid blocking the main thread
tts_queue = queue.Queue()

# Thread to handle text-to-speech with cross-platform support
def tts_worker():
    while True:
        text = tts_queue.get()
        if text == "STOP":
            break

        if tts_engine['type'] == 'pyttsx3':
            try:
                tts_engine['engine'].say(text)
                tts_engine['engine'].runAndWait()
            except Exception as e:
                # Just silently continue on error - no need to crash the thread
                pass
        elif tts_engine['type'] == 'gtts':
            try:
                tts_engine['engine'](text)
            except Exception as e:
                # Just silently continue on error
                pass

        tts_queue.task_done()

# Start TTS thread if engine is available
if tts_engine['type'] != 'none':
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
else:
    # If no TTS is available, create a simple function to store the text that would have been spoken
    def mock_tts_store(text):
        st.session_state.last_spoken = text

    tts_queue.put = mock_tts_store

# Video processor class for WebRTC with improved error handling
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_counter = 0
        self.last_emotion = "Detecting..."
        self.processing_active = model_best is not None and face_cascade is not None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Skip processing if resources aren't available
        if not self.processing_active:
            cv2.putText(img, "Model or face detector not loaded",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            self.frame_counter += 1

            # Process each detected face
            for (x, y, w, h) in faces:
                # Process every 30 frames to reduce load
                if self.frame_counter % 30 == 0:
                    # Extract the face region
                    face_roi = img[y:y + h, x:x + w]

                    # Resize the face image to the required input size for the model
                    face_image = cv2.resize(face_roi, (48, 48))
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image = np.expand_dims(face_image, axis=0)
                    face_image = np.expand_dims(face_image, axis=-1)  # Add channel dimension

                    # Predict emotion using the loaded model
                    predictions = model_best.predict(face_image)
                    self.last_emotion = class_names[np.argmax(predictions)]

                    # Store emotion in history using session state
                    append_to_emotion_history(self.last_emotion)

                    # Speak the emotion without blocking
                    tts_queue.put(f"emotion is {self.last_emotion}")

                # Draw a rectangle around the face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Display the emotion label on the frame
                cv2.putText(img, f"Emotion: {self.last_emotion}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            # Handle any errors during processing
            error_msg = f"Error: {str(e)}"
            cv2.putText(img, error_msg, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main app
def main():
    st.title("Real Time Face Emotion Detection Application for Blind Therapists")

    activities = ["Home", "Webcam Face Detection", "Emotion Trend Analysis", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                            <h4 style="color:white;text-align:center;">
                            Start web cam and check for real time facial emotions.</h4>
                            </div>
                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has the following functionalities:
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognition.
                 3. Text to speech conversion.
                 4. Emotion trend analysis.
                 """)

        # Display deployment information
        st.info("""
        **Deployment Notice**:
        - This application requires webcam access.
        - When using for the first time, your browser will ask for camera permissions.
        - Text-to-speech functionality may be limited in cloud deployment.
        """)

        # Show TTS status
        if tts_engine['type'] == 'none':
            st.warning("Text-to-speech is not available in this environment. Emotions will only be displayed visually.")
        else:
            st.success(f"Text-to-speech is available using {tts_engine['type']} engine.")

    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on 'Start' to begin the webcam stream.")

        # Check if resources are loaded
        if model_best is None or face_cascade is None:
            st.error("Required resources (model or face detector) could not be loaded. Please check the application logs.")
            return

        # Updated webrtc_streamer call
        webrtc_ctx = webrtc_streamer(
            key="fer",
            video_processor_factory=VideoProcessor,
            async_processing=True,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        # Display instructions
        if webrtc_ctx.state.playing:
            st.info("Webcam is active. The application is now analyzing facial emotions.")

            # Show last detected emotion as text for accessibility
            if 'last_spoken' in st.session_state:
                st.subheader("Last Detected Emotion")
                st.write(st.session_state.last_spoken)

    elif choice == "Emotion Trend Analysis":
        st.header("Emotion History Analysis")

        emotion_history = get_emotion_history()
        if not emotion_history:
            st.warning("No emotion data recorded yet. Use the webcam feature first to collect data.")
        else:
            # Count occurrences of each emotion
            emotion_counts = Counter(emotion_history)

            # Create bar chart
            st.subheader("Emotion Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())

            ax.bar(emotions, counts, color='skyblue')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Detected Emotions')
            plt.xticks(rotation=45)

            st.pyplot(fig)

            # Add timeline view
            st.subheader("Emotion Timeline")

            # Convert emotion categories to numbers for plotting
            emotion_to_num = {emotion: i for i, emotion in enumerate(class_names)}
            emotion_nums = [emotion_to_num.get(e, 0) for e in emotion_history]

            # Plot the timeline
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(range(len(emotion_history)), emotion_nums, marker='o', linestyle='-')
            ax2.set_yticks(range(len(class_names)))
            ax2.set_yticklabels(class_names)
            ax2.set_xlabel('Detection Sequence')
            ax2.set_ylabel('Emotion')
            ax2.set_title('Emotion Timeline')
            ax2.grid(True)

            st.pyplot(fig2)

            # Summary statistics
            st.subheader("Summary")
            most_common = emotion_counts.most_common(1)[0][0]
            st.write(f"Most common emotion: **{most_common}**")
            st.write(f"Total emotions recorded: **{len(emotion_history)}**")

            # Add session data reset button
            if st.button("Reset Emotion History"):
                st.session_state.emotion_history = []
                st.success("Emotion history has been reset.")
                st.experimental_rerun()

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                             <h4 style="color:white;text-align:center;">
                             Real time face emotion detection application using TensorFlow and WebRTC.</h4>
                             </div>
                             </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                     <div style="background-color:#98AFC7;padding:10px">
                     <h4 style="color:white;text-align:center;">This Application is developed using Streamlit Framework, WebRTC, TensorFlow and Keras library for demonstration purpose.</h4>
                     <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                     </div>
                     <br></br>
                     <br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)

if __name__ == "__main__":
    main()