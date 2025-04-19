import cv2
import numpy as np
import streamlit as st
import json
import os
import urllib.request
import pyttsx3
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def append_to_json(file_name, data):
    try:
        # Load existing data
        with open(file_name, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, start with an empty list
        existing_data = []

    # Append new data
    existing_data.append(data)

    # Write back to file
    with open(file_name, 'w') as f:
        json.dump(existing_data, f)
        
def read_from_json(file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_webcam():
    # Try different camera indices
    for i in range(3):  # Try indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
    
    # If no camera works, create a dummy frame for testing
    st.warning("No camera found. Using a dummy frame for testing.")
    dummy_cap = type('obj', (object,), {
        'read': lambda self: (True, np.ones((480, 640, 3), dtype=np.uint8) * 255),
        'isOpened': lambda self: True,
        'release': lambda self: None
    })()
    return dummy_cap, -1

def ensure_cascade_file(file_path='haarcascade_frontalface_default.xml'):
    """Ensure the cascade file exists, download if not"""
    if not os.path.isfile(file_path):
        st.info(f"Downloading {file_path}...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, file_path)
            st.success(f"Successfully downloaded {file_path}")
        except Exception as e:
            st.error(f"Failed to download {file_path}: {e}")
            return None
    return cv2.CascadeClassifier(file_path)

def load_emotion_model(model_path='Streamlit_FER_system-main/face_model.h5'):
    """Load the emotion detection model with error handling"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        st.info(f"Please make sure '{model_path}' exists in the current directory")
        return None

def main():
    st.title("Real Time Face Emotion Detection Application for Blind Therapists")
    
    # Sidebar for navigation
    activities = ["Home", "Webcam Face Detection", "Emotion Trend Analysis", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    # Initialize text-to-speech engine
    try:
        engine = pyttsx3.init()
    except Exception as e:
        st.warning(f"Text-to-speech engine could not be initialized: {e}")
        engine = None
    
    # Class names for emotions
    class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Load cascade classifier
    face_cascade = ensure_cascade_file()
    if face_cascade is None:
        st.error("Failed to load face detection cascade classifier. Application may not work properly.")
    
    # Load emotion model
    model = load_emotion_model()
    if model is None:
        st.error("Failed to load emotion model. Please check if the model file exists.")
    
    # Home page
    if choice == "Home":
        st.markdown("""
        <div style="background-color:#6D7B8D;padding:10px">
            <h4 style="color:white;text-align:center;">
            Start web cam and check for real time facial emotions.</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("""
        The application has the following functionalities:
        1. Real time face detection using web cam feed.
        2. Real time face emotion recognition.
        3. Text to speech conversion.
        4. Emotion trend analysis over time.
        """)
    
    # Webcam Face Detection
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        
        # Create placeholders
        frame_window = st.empty()
        status_text = st.empty()
        
        # Start webcam automatically
        cap, camera_idx = get_webcam()
        if camera_idx >= 0:
            status_text.success(f"Using camera index: {camera_idx}")
        
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    status_text.warning("Failed to capture frame from webcam. Check your camera connection.")
                    break
                
                try:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    
                    # Process each detected face
                    for (x, y, w, h) in faces:
                        frame_counter += 1
                        
                        # Extract and preprocess the face region
                        face_roi = frame[y:y+h, x:x+w]
                        face_img = cv2.resize(face_roi, (48, 48))
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        face_img = image.img_to_array(face_img)
                        face_img = np.expand_dims(face_img, axis=0)
                        
                        # Predict emotion
                        if model is not None:
                            predictions = model.predict(face_img)
                            emotion_label = class_names[np.argmax(predictions)]
                            
                            # Save emotion to history
                            append_to_json('emotions.json', emotion_label)
                            
                            # Draw emotion label and face rectangle
                            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            
                            # Speak the emotion (every 10 frames)
                            if frame_counter % 10 == 0 and engine is not None:
                                engine.say(f"emotion is {emotion_label}")
                                engine.runAndWait()
                    
                    # Display the frame
                    frame_window.image(frame, channels="BGR")
                    
                    # Optional: Add a stop button
                    if st.button("Stop Camera", key="stop"):
                        break
                        
                except Exception as e:
                    status_text.error(f"Error processing frame: {str(e)}")
                    break
                    
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        
        finally:
            cap.release()
            status_text.info("Camera stopped")
    
    # Emotion Trend Analysis
    elif choice == "Emotion Trend Analysis":
        st.header("Emotion Trend Analysis")
        
        # Load emotion history
        emotion_history = read_from_json('emotions.json')
        
        if not emotion_history:
            st.warning("No emotion data found. Please run the webcam detection first.")
        else:
            # Count occurrences of each emotion
            emotion_counts = Counter(emotion_history)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            ax.bar(emotions, counts, color='skyblue')
            ax.set_title('Emotion Distribution')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Count')
            
            # Display the chart
            st.pyplot(fig)
            
            # Show emotion percentages
            st.subheader("Emotion Percentages")
            total = sum(counts)
            for emotion, count in emotion_counts.items():
                percentage = (count / total) * 100
                st.write(f"{emotion}: {percentage:.2f}%")
            
            # Show most frequent emotion
            most_common = emotion_counts.most_common(1)[0][0]
            st.subheader(f"The most frequent emotion detected was: {most_common}")
    
    # About page
    elif choice == "About":
        st.subheader("About this app")
        st.markdown("""
        <div style="background-color:#6D7B8D;padding:10px">
            <h4 style="color:white;text-align:center;">
            Real time face emotion detection application using TensorFlow.</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#98AFC7;padding:10px">
            <h4 style="color:white;text-align:center;">This Application is developed using Streamlit Framework, OpenCV, TensorFlow and Keras library for demonstration purpose.</h4>
            <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
