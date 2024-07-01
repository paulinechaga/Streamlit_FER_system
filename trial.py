import cv2
import numpy as np
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import pyttsx3
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import json

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
model_best = load_model('face_model.h5')
engine = pyttsx3.init()

class_names = ['Angry', 'Disgusted', 'Fear',
               'Happy', 'Sad', 'Surprise', 'Neutral']
# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)
frame_window = st.image([])
frame_counter = 0
emotion_history=[]

st.title("Real Time Face Emotion Detection Application for Blind therapists")
activities = ["Home", "Webcam Face Detection", "Emotion Trend Analysis","About"]  # Add Emotion Trend Analysis to the activities
choice = st.sidebar.selectbox("Select Activity", activities)
if choice == "Home":
    html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Start web cam and check for real time facial emotions.</h4>
                                        </div>
                                        </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    st.write("""
             The application has the following functionalities.
             1. Real time face detection using web cam feed.
             2. Real time face emotion recognization.
             3. Text to speech conversion.
             """)

elif choice == "Webcam Face Detection":
     st.header("Webcam Live Feed")
     while True:
        ret, frame = cap.read()
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            frame_counter += 1
            print(frame_counter)
            if frame_counter % 1 == 0:
                # Extract the face region
                face_roi = frame[y:y + h, x:x + w]
                # Resize the face image to the required input size for the model
                face_image = cv2.resize(face_roi, (48, 48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = image.img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_image = np.vstack([face_image])
                # Predict emotion using the loaded model
                if model_best is not None:
                    predictions = model_best.predict(face_image)
                emotion_label = class_names[np.argmax(predictions)]
                # with open ("emotion_history.json","w") as f:
                #     f.write()
                append_to_json('emotions.json', emotion_label)

                #emotion_history.append(emotion_label) 
                
                # Display the emotion label on the frame
                cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # speak the emotion

                engine.say("emotion is" + emotion_label)
                engine.runAndWait()
        # Display the resulting frame
        frame_window.image(frame, channels="BGR")
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
                # engine.say("emotion is" + emotion_label)
                # engine.runAndWait()
            # break
            # Release the webcam and close all windows
                cap.release()
                cv2.destroyAllWindows()

   

elif choice == "Emotion Trend Analysis":
    # Count the occurrences of each emotion
    emotion_history = read_from_json('emotions.json')
    emotion_counts = Counter(emotion_history)
    print(emotion_counts)

    # # Prepare data for radar chart
    # labels = np.array(list(emotion_counts.keys()))
    # counts = np.array(list(emotion_counts.values()))
    # num_vars = len(labels)

    # # Compute angle for each axis in the plot
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # # The radar chart will be a closed polygon, so we need to "complete the loop"
    # # and append the start value to the end.
    # counts = np.concatenate((counts, [counts[0]]))
    # angles += angles[:1]

    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    # ax.fill(angles, counts, color='blue', alpha=0.25)
    # ax.set_yticklabels([])
    # ax.set_thetagrids(np.degrees(angles), labels)

    # st.pyplot(fig)


elif choice == "About":
    st.subheader("About this app")
    html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Real time face emotion detection application using tensorflow.</h4>
                                </div>
                                </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)
    html_temp4 = """
                         		<div style="background-color:#98AFC7;padding:10px">
                         		<h4 style="color:white;text-align:center;">This Application is developed using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.  </h4>
                         		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                         		</div>
                         		<br></br>
                         		<br></br>"""
    st.markdown(html_temp4, unsafe_allow_html=True)
else:
    pass


