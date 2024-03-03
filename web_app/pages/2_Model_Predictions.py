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
st.title('Face Recognition App')

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
model = loadmodel('./saved_models/siamese_model_last.h5')

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

    verification_images_folder = './application_data/verification_images'
    for folder in os.listdir(verification_images_folder):
        folder_path = os.path.join(verification_images_folder, folder)
        if not os.path.isdir(folder_path):  # Skip if it's not a directory
            continue
        for image_file in os.listdir(folder_path):
            if image_file.startswith('.'):  # Skip hidden files
                continue
            image_path = os.path.join(folder_path, image_file)
            input_img = preprocess_image('./application_data/input_image/captured_image.jpg')
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

    min_score_index = np.argmin(results)
    name_with_min_score = names[min_score_index]

    return results, verified, name_with_min_score


# open webcam

# def main():
#     st.caption("Powered by OpenCV, Streamlit")
#     open_camera = st.button("Open Camera")

#     if open_camera:
#         cascPath = './haarcascade_frontalface_default.xml'
#         faceCascade = cv2.CascadeClassifier(cascPath)

#         cap = cv2.VideoCapture(0)

#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             st.image(frame, channels="RGB")

#             image_folder = "./application_data/input_image"
#             if not os.path.exists(image_folder):
#                 os.makedirs(image_folder)
#             image_path = os.path.join(image_folder, "captured_image.jpg")
#             cv2.imwrite(image_path, frame)
#             st.write("Image captured and saved!")

#             # input_image = preprocess_image(image_path)
#             if image_path is not None:
#                 # Perform verification on the input image
#                 results, verified, min_score_name = verify_image(model, 0.9, 0.4)
#                 st.write(f"Verified: {verified}")
#                 st.write(f"Welcome {min_score_name}")

#         cap.release()

# if __name__ == "__main__":
#     main()

# def main():
#     st.caption("Powered by OpenCV, Streamlit")
#     open_camera = st.button("Open Camera")

#     if open_camera:
#         cascPath = './haarcascade_frontalface_default.xml'
#         faceCascade = cv2.CascadeClassifier(cascPath)

#         video_capture = cv2.VideoCapture(0)

#         while True:
#             ret, frame = video_capture.read()

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.1,
#                 minNeighbors=5,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )

#             # Crop and display the face region
#             for (x, y, w, h) in faces:
#                 face_frame = frame[y:y+h, x:x+w]
#                 face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
#                 st.image(face_frame, channels="RGB")

#                 image_folder = "./application_data/input_image"
#                 if not os.path.exists(image_folder):
#                     os.makedirs(image_folder)
#                 image_path = os.path.join(image_folder, "captured_image.jpg")
#                 cv2.imwrite(image_path, frame)
#                 st.write("Image captured and saved!")

#                 # Perform verification on the face region
#                 if image_path is not None:
#                     # Perform verification on the input image
#                     results, verified, min_score_name = verify_image(model, 0.6, 0.4)
#                     st.write(f"Verified: {verified}")
#                     st.write(f"Welcome {min_score_name}")

#             cv2.imshow('Video', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         video_capture.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

def main():
    st.caption("Powered by OpenCV, Streamlit")
    open_camera = st.button("Open Camera")

    if open_camera:
        cascPath = './haarcascade_frontalface_default.xml'
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
                image_folder = "./application_data/input_image"
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, "captured_image.jpg")
                cv2.imwrite(image_path, face_frame)
                st.write("Image captured and saved!")

                # input_image = preprocess_image(image_path)
                if image_path is not None:
                    # Perform verification on the input image
                    results, verified, min_score_name = verify_image(model, 0.7, 0.4)
                    st.write(f"Verified: {verified}")
                    st.write(f"Welcome {min_score_name}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()