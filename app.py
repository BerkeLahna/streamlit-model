import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json

# Load the Keras model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define MediaPipe hands and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess hand landmarks
landmark_buffer = []

# Function to preprocess hand landmarks
def preprocess_landmarks(landmarks):
    # Extract the x, y, and z coordinates of each landmark and flatten them into a numeric array
    processed_landmarks = []
    for landmark in landmarks:
        processed_landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return processed_landmarks

image_placeholder = st.empty()
text_placeholder = st.empty()

cap = cv2.VideoCapture(0)  # Use your desired video source, e.g., a video file
output_text = ""

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert the frame to RGB (MediaPipe requires RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Preprocess the hand landmarks
                processed_landmarks = preprocess_landmarks(hand_landmarks.landmark)

                # Add the processed landmarks to the buffer
                landmark_buffer.append(processed_landmarks)

                # Maintain a buffer of 30 landmarks
                if len(landmark_buffer) > 30:
                    landmark_buffer.pop(0)  # Remove the oldest landmarks

                # If the buffer is full, pass it through your Keras model
                if len(landmark_buffer) == 30:
                    # Reshape the processed landmarks to match the model's expected input shape
                    processed_landmarks = np.array(landmark_buffer)
                    
                    # Pass the processed landmarks through your Keras model
                    prediction = model.predict(processed_landmarks.reshape(1, 30, 63))
                    predicted_label = np.argmax(prediction)
                    predicted_probability = prediction[0][predicted_label]

                    # Set a threshold for the probability (e.g., 50%)
                    probability_threshold = 0.5

                    if predicted_probability >= probability_threshold:
                        if predicted_label == 0:
                            output_text = f"Output: a ({predicted_probability * 100:.2f}%)"
                            text_placeholder.text(output_text)
                        elif predicted_label == 1:
                            output_text = f"Output: b ({predicted_probability * 100:.2f}%)"
                            text_placeholder.text(output_text)

                        elif predicted_label == 2:
                            output_text = f"Output: c ({predicted_probability * 100:.2f}%)"
                            text_placeholder.text(output_text)

                    else:
                        output_text = "Output: Unknown"

        # Draw landmarks on the frame even if no hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the output on the frame
        cv2.rectangle(frame, (300, 0), (600, 40), (245, 117, 16), -1)
        cv2.putText(frame, output_text, (305, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame using Streamlit
        # st.image(frame, channels="BGR", use_column_width=True)
        image_placeholder.image(frame, channels="BGR", use_column_width=True)




# Rest of your Streamlit app




# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
