# import dependencies
import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras import backend as K
import cv2
import os
import time


# page title 
st.title('Facial Recognition App')

#import loss function for custom objects
def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - (y_pred), 0))
        return K.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

# Load model
@st.cache_data()
def loadmodel(file):
    model = keras.models.load_model(file, custom_objects={'contrastive_loss': loss}, compile=False)
    return model

# instantiate model
model = loadmodel('../models/model.h5')

# st.sidebar.title('Menu')

def preprocess_image(image_path, target_size=(256,256)):
    """Preprocesses the given image.

    Arguments:
        image_path: The path to the image file.
        target_size: A tuple of integers indicating the target size.

    Returns:
        A preprocessed image.
    """
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Check if the image was loaded successfully
    if image is None:
        print("Failed to load image:", image_path)
        return None

    # Step 3: Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize the grayscale image
    resized_image = cv2.resize(gray_image, target_size)

    # Step 5: Normalize pixel values
    normalized_image = resized_image / 255.0

    return normalized_image

def verify_image(model, detection_threshold, verification_threshold):
    results = []
    names = []

    verification_images_folder = '../web_app/application_data/verification_images'

    # Preprocess the input image
    input_img = preprocess_image('../web_app/application_data/input_image/captured_image.jpg')

    for folder in os.listdir(verification_images_folder):
        folder_path = os.path.join(verification_images_folder, folder)
        if not os.path.isdir(folder_path):  # Skip if it's not a directory
            continue
        for image_file in os.listdir(folder_path):
            if image_file.startswith('.'):  # Skip hidden files
                continue
            image_path = os.path.join(folder_path, image_file)
            validation_img = preprocess_image(image_path)

            # Check if input_img and validation_img have the same shape
            if input_img.shape != validation_img.shape:
                # Handle the mismatched shape
                continue

            print(f"Input Image Shape: {input_img.shape}")
            print(f"Validation Image Shape: {validation_img.shape}")

            result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            print(f"Prediction Result: {result}")
            results.append(result)
            names.append(os.path.splitext(folder)[0])

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold

    max_score_index = np.argmax(results)
    name_with_max_score = names[max_score_index]

    # min_score_index = np.argmin(results)
    # name_with_min_score = names[min_score_index]

    return results, verified, name_with_max_score


def main():
    st.caption("Powered by OpenCV, Streamlit")
    
    # Option to open camera or upload an image
    option = st.radio("Choose an option:", ["Open Camera", "Upload Image"])

    if option == "Open Camera":
        open_camera = st.button("Open Camera")
        verification_done = False

        if open_camera and not verification_done:
            cascPath = '../data/haarcascade_frontalface_default.xml'
            faceCascade = cv2.CascadeClassifier(cascPath)

            video_capture = cv2.VideoCapture(0)

            while True:
                ret, frame = video_capture.read()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB")

                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Crop and display the face region
                for (x, y, w, h) in faces:
                    face_frame = frame[y:y+h, x:x+w]
                    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                    st.image(face_frame, channels="BGR")

                    # Perform verification on the face region
                    image_folder = "../web_app/application_data/input_image"
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                    image_path = os.path.join(image_folder, "captured_image.jpg")
                    cv2.imwrite(image_path, face_frame)
                    st.write("Image captured and saved.")

                    # input_image = preprocess_image(image_path)
                    if image_path is not None:
                        # Perform verification on the input image
                        results, verified, min_score_name = verify_image(model, 0.03, 0.03)
                        if verified:
                            st.write(f"You are Verified!")
                            st.write(f"Welcome {min_score_name}")
                        else:
                            st.write(f"Unknown User")
                        verification_done = True  # Set the flag to indicate verification is done
                        break  # Exit the loop after verification is done

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
    else:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            st.image(image, channels="RGB")

            # Perform verification on the input image
            image_path = "../web_app/application_data/input_image/uploaded_image.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            st.write("Image uploaded and saved.")

            # input_image = preprocess_image(image_path)
            if image_path is not None:
                # Perform verification on the input image
                results, verified, min_score_name = verify_image(model, 0.134, 0.18175)
                if verified:
                    st.write(f"You are verified!")
                    st.write(f"Welcome {min_score_name}")
                else:
                    st.write(f"Unknown User")

if __name__ == "__main__":
    main()