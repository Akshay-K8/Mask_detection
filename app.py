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
    st.title("Mask Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Preprocess the image and get cropped face(s)
        cropped_faces = preprocess_image(uploaded_file)

        for face in cropped_faces:
            # Display the cropped face image
            st.image(face, channels="BGR", use_column_width=True)

            # Convert the NumPy array to a file-like object
            face_bytes = BytesIO()
            image = cv2.imencode('.jpg', face)[1].tobytes()
            face_bytes.write(image)
            face_bytes.seek(0)  # Reset the pointer to the beginning of the file

            # Predict on the cropped face
            with st.spinner('Processing...'):
                prediction = predict(face_bytes)
            st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()