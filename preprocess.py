import cv2
import os
import numpy as np
from PIL import Image

# Path to the directory containing images
input_dir = "input_images"
output_dir = "processed_images"
resolusi = 50 #Pixel

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Path to the prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file
)

def detect_faces_dnn(image):
    # Prepare the input blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)

    # Perform detection
    detections = face_net.forward()
    h, w = image.shape[:2]

    # Extract detected faces
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))  # Convert to x, y, w, h format

    return faces

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return False

    faces = detect_faces_dnn(image)
    if len(faces) == 0:
        print(f"No face detected with DNN in {image_path}. Skipping.")
        return False

    x, y, w, h = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h)
    half_size = crop_size // 2

    crop_x1 = max(center_x - half_size, 0)
    crop_y1 = max(center_y - half_size, 0)
    crop_x2 = min(crop_x1 + crop_size, image.shape[1])
    crop_y2 = min(crop_y1 + crop_size, image.shape[0])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_face = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_face = cv2.resize(cropped_face, (resolusi, resolusi), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized_face)
    print(f"Processed image saved to {output_path}")
    return True

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.heic')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.split('.')[0] + '.png')
        preprocess_image(input_path, output_path)
