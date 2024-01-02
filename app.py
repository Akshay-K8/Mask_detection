import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io

# Load the trained model
model = load_model('./mask_detection_model.h5')  # Load your trained model

def predict(image_file):
    img = image.load_img(image_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make predictions
    prediction = model.predict(img_array)

    # Interpret the prediction
    if prediction[0][0] > 0.5:  # Assuming binary classification (mask/no-mask)
        return "With Mask"
    else:
        return "Without Mask"

def main():
    st.title("Mask Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        prediction = predict(uploaded_file)
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
