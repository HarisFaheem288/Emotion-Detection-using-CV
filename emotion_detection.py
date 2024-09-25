
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('emotion_detection_model.keras')

# Emotion labels based on your training data
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Streamlit app title
st.title("Emotion Detection App")

# Option for image upload
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Option for capturing a single frame from webcam
if st.button('Capture from webcam'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please check if it's connected.")
    else:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from the webcam.")
        else:
            # Convert captured frame (BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            uploaded_image = Image.fromarray(frame_rgb)
            st.image(uploaded_image, caption="Captured Image", use_column_width=True)
        cap.release()

# Process the image if uploaded or captured
if uploaded_image is not None:
    # Open the uploaded image if not from webcam
    if isinstance(uploaded_image, Image.Image):
        image = uploaded_image
    else:
        # Open the uploaded image
        image = Image.open(uploaded_image)

    # Display the image in the Streamlit app
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to grayscale
    image_gray = image.convert('L')  # Convert to grayscale

    # Resize to 48x48 pixels, which is the input size expected by the model
    image_resized = image_gray.resize((48, 48))

    # Normalize the image (0-255 to 0-1)
    image_normalized = np.array(image_resized) / 255.0

    # Reshape to match the model input shape (1, 48, 48, 1)
    image_reshaped = np.reshape(image_normalized, (1, 48, 48, 1))

    # Predict the emotion
    prediction = model.predict(image_reshaped)
    emotion_index = np.argmax(prediction[0])
    emotion = emotion_labels[emotion_index]

    # Display the predicted emotion
    st.write(f"Predicted Emotion: {emotion}")

# Add a footer or additional information if needed
st.write("Developed by [Your Name]. Feel free to explore and give feedback!")
