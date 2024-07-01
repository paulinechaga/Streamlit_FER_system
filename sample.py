import cv2
import numpy as np
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import pyttsx3
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import os
from math import pi
import pandas as pd

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
emotion_history = []  # Add this line to keep track of emotions

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
    if os.path.exists("emotion.txt"):
        os.remove("emotion.txt")
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

                with open("emotion.txt", 'a') as f:
                    f.write(emotion_label + "\n")
                        
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
    with open("emotion.txt", 'r') as f:
        for line in f:
            # emotion = f.readline().strip()
            emotion_history.append(line.strip())
    emotion_counts = Counter(emotion_history)
    st.write(emotion_counts)
    df = pd.DataFrame.from_dict(emotion_counts, orient='index').reset_index()
    df.columns = ['Emotion', 'Count']

    # Compute angle each axis in the plot will have.
    angles = [n / float(df['Emotion'].size) * 2 * pi for n in range(df['Emotion'].size)]
    angles += angles[:1]

    # Initialise the radar plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], df['Emotion'], color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)
    plt.ylim(0, df['Count'].max())

    # Plot data
    ax.plot(angles, df['Count'].tolist()+df['Count'].tolist()[:1])

    # Fill area
    ax.fill(angles, df['Count'].tolist()+df['Count'].tolist()[:1], 'b', alpha=0.1)

    # Display the plot in Streamlit
    st.pyplot(plt)
    # fig,ax=plt.subplots()
    # ax.bar(emotion_counts, bins=7,height=20)
    # st.pyplot(fig)
    # emotion_counts.keys(), emotion_counts.values()
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


