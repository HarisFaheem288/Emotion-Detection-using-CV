import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64

# Load the trained model
model = load_model('emotion_detection_model.keras')

# Emotion labels based on your training data
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Streamlit app title
st.title("Emotion Detection App")

# Instructions
st.write("Please upload an image or capture one using your webcam.")

# Option for image upload
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Web-based webcam capture
st.write("Or capture an image from your webcam:")

webcam_capture_html = """
    <script>
    function captureImage() {
        var video = document.querySelector("#videoElement");
        var canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        var dataURL = canvas.toDataURL('image/png');
        streamlit.setComponentValue(dataURL);
    }

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            var video = document.querySelector("#videoElement");
            video.srcObject = stream;
        })
        .catch(function(error) {
            console.log("Unable to access the webcam: ", error);
        });
    </script>

    <video autoplay="true" id="videoElement" style="width: 100%; height: auto;"></video>
    <button onclick="captureImage()">Capture Image</button>
"""

# Display the HTML and JavaScript for the webcam
st.markdown(webcam_capture_html, unsafe_allow_html=True)

# Capture the base64 string from JavaScript
captured_image = st.experimental_get_query_params().get('dataURL', None)

# If there's a captured image
if captured_image:
    # Convert the base64 image to a format that can be processed
    img_data = base64.b64decode(captured_image.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    st.image(img, caption="Captured Image", use_column_width=True)
    uploaded_image = img

# Process the image if uploaded or captured
if uploaded_image is not None:
    # Convert the image to grayscale
    image_gray = uploaded_image.convert('L')  # Convert to grayscale

    # Resize to 48x48 pixels, which is the input size expected by the model
    image_resized = image_gray.resize((48, 48))

    # Normalize the image (from 0-255 to 0-1)
    image_normalized = np.array(image_resized) / 255.0

    # Reshape to match the model input shape (1, 48, 48, 1)
    image_reshaped = np.reshape(image_normalized, (1, 48, 48, 1))

    # Predict the emotion
    prediction = model.predict(image_reshaped)
    emotion_index = np.argmax(prediction[0])
    emotion = emotion_labels[emotion_index]

    # Display the predicted emotion
    st.write(f"Predicted Emotion: {emotion}")

# Footer or additional information
st.write("Developed by [Your Name]. Feel free to explore and give feedback!")
