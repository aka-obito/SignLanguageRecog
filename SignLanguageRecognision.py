import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = "sign_language_cnn_model.keras"  # Update with your actual model file
model = load_model(MODEL_PATH)

# Define class labels (adjust as per your trained model)
class_labels = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
    17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# Function to preprocess the image for the model
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    resized = cv2.resize(gray, (28, 28))
    array = img_to_array(resized)
    array = np.expand_dims(array, axis=0)
    array /= 255.0
    return array

# Streamlit UI
def main():
    st.title("Real-time Sign Language Recognition")
    st.write("This application recognizes sign language gestures using a webcam.")

    # Display the reference sign language chart
    try:
        st.sidebar.image("Sign.png", caption="Sign Language Chart", use_container_width=True)
    except Exception as e:
        st.sidebar.write("Error loading sign language chart:", str(e))

    # Start and Stop buttons
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    FRAME_WINDOW = st.image([])

    cap = None  # Initialize the webcam object

    # Webcam loop
    if start_button:
        cap = cv2.VideoCapture(0)
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Flip and define region of interest (ROI)
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            top, right, bottom, left = 50, width - 400, 200, width - 50
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

            roi = frame[top:bottom, right:left]

            if roi.size > 0:
                processed = preprocess_image(roi)
                prediction = model.predict(processed)
                predicted_class = np.argmax(prediction)
                label = class_labels.get(predicted_class, "Unknown")

                # Display recognized letter
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if stop_button and cap is not None:
        cap.release()

if __name__ == "__main__":
    main()
