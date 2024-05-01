import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from io import BytesIO

# Load the trained model
model = load_model('./maskdetection.h5')

def predict(image_data):
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:  # Assuming binary classification (mask/no-mask)
        return "With Mask"
    else:
        return "Without Mask"

def preprocess_image(image_file):
    # Convert the file object to bytes
    image_bytes = image_file.read()
    # Create a NumPy array from the bytes
    img_array = np.frombuffer(image_bytes, np.uint8)
    # Decode the NumPy array into an OpenCV-compatible format
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Crop the image to include only the face(s)
    cropped_images = []
    for (x, y, w, h) in faces:
        cropped_image = img[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
    return cropped_images

def main():
    st.set_page_config(page_title="Mask Detection App", page_icon=":mask:")

    # Set background color
    page_bg_color = """
    <style>
    body {
        background-color: #f0f0f0;
    }
    </style>
    """
    st.markdown(page_bg_color, unsafe_allow_html=True)

    # Set app title
    st.title("🚫 Mask Detection App")

    # Create option selection
    option = st.radio("Select an option", ["Upload Image", "Capture from Webcam"], index=0, horizontal=True)

    if option == "Upload Image":
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            cropped_faces = preprocess_image(uploaded_file)
            if not cropped_faces:
                st.warning("No faces detected in the uploaded image.")
            else:
                for face in cropped_faces:
                    st.image(face, channels="BGR", use_column_width=True)
                    face_bytes = BytesIO()
                    image = cv2.imencode('.jpg', face)[1].tobytes()
                    face_bytes.write(image)
                    face_bytes.seek(0)
                    with st.spinner('Processing...'):
                        prediction = predict(face_bytes)
                    st.success(f"Prediction: {prediction}", icon="🧍‍♀️")

    elif option == "Capture from Webcam":
        st.subheader("Capture from Webcam")
        capture = st.camera_input("Take a picture")
        if capture is not None:
            img = cv2.imdecode(np.frombuffer(capture.read(), np.uint8), cv2.IMREAD_COLOR)
            cropped_faces = preprocess_image(BytesIO(cv2.imencode('.jpg', img)[1].tobytes()))
            if not cropped_faces:
                st.warning("No faces detected in the captured image.")
            else:
                for face in cropped_faces:
                    st.image(face, channels="BGR", use_column_width=True)
                    face_bytes = BytesIO()
                    image = cv2.imencode('.jpg', face)[1].tobytes()
                    face_bytes.write(image)
                    face_bytes.seek(0)
                    with st.spinner('Processing...'):
                        prediction = predict(face_bytes)
                    st.success(f"Prediction: {prediction}", icon="🧍‍♀️")

if __name__ == '__main__':
    main()
