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

def process_image(image_file):
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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_bytes = BytesIO()
        face_image = cv2.imencode('.jpg', face)[1].tobytes()
        face_bytes.write(face_image)
        face_bytes.seek(0)
        prediction = predict(face_bytes)
        
        # Draw rectangle
        color = (0, 255, 0) if prediction == "With Mask" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
        
        # Put text
        cv2.putText(img, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    
    return img

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
    st.title("😷 Mask Detection App")
    # Create option selection
    option = st.radio("Select an option", ["Upload Image", "Capture from Webcam"], index=0, horizontal=True)
    
    if option == "Upload Image":
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            processed_image = process_image(uploaded_file)
            st.image(processed_image, channels="BGR", use_column_width=True)
    
    elif option == "Capture from Webcam":
        st.subheader("Capture from Webcam")
        capture = st.camera_input("Take a picture")
        if capture is not None:
            processed_image = process_image(capture)
            st.image(processed_image, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()
