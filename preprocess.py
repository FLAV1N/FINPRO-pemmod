import cv2
import os
import numpy as np
from PIL import Image

# Path to the directory containing images
input_dir = "input_images"  # Replace with the path to your input images
output_dir = "processed_images"  # Directory to save processed images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to convert HEIC to an OpenCV-readable format
def load_image(image_path):
    if image_path.lower().endswith('.heic'):
        heif_file = pyheif.read(image_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(image_path)

# Function to preprocess an image
def preprocess_image(image_path, output_path):
    # Read the image
    image = load_image(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return False

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return False

    # Take the first detected face (assuming only one face is required)
    x, y, w, h = faces[0]

    # Calculate the center of the face and ensure a 1:1 ratio crop
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h)  # Ensure the cropped area is square
    half_size = crop_size // 2

    # Define the square crop boundaries
    crop_x1 = max(center_x - half_size, 0)
    crop_y1 = max(center_y - half_size, 0)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Ensure the crop stays within the image boundaries
    crop_x2 = min(crop_x2, image.shape[1])
    crop_y2 = min(crop_y2, image.shape[0])

    # Crop the image
    cropped_face = gray[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to 50x50
    resized_face = cv2.resize(cropped_face, (50, 50), interpolation=cv2.INTER_AREA)

    # Save the processed image
    cv2.imwrite(output_path, resized_face)
    print(f"Processed image saved to {output_path}")
    return True

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.heic')):  # Added HEIC support
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.split('.')[0] + '_processed.png')  # Save as JPEG
        preprocess_image(input_path, output_path)
