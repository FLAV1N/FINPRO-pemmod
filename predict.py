import sys
import cv2
import numpy as np
import pickle
from bpnn_v2 import sigmoid  # Import sigmoid function from the training code

# Load the DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Path to the prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file
)

def softmax(logits, temperature=0.2):  # Lower temperature increases confidence
    exp_logits = np.exp((logits - np.max(logits)) / temperature)
    return exp_logits / np.sum(exp_logits)

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

def preprocess_and_predict(image_path, model_path="bpnn_model_best.pkl"):
    # Load the model
    with open(model_path, "rb") as f:
        params = pickle.load(f)

    # Preprocess the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return

    # Detect faces using DNN
    faces = detect_faces_dnn(image)
    if len(faces) == 0:
        print("No face detected with DNN.")
        return

    # Use the first detected face
    x, y, w, h = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h)
    half_size = crop_size // 2

    # Define square crop boundaries
    crop_x1 = max(center_x - half_size, 0)
    crop_y1 = max(center_y - half_size, 0)
    crop_x2 = min(crop_x1 + crop_size, image.shape[1])
    crop_y2 = min(crop_y1 + crop_size, image.shape[0])

    # Convert to grayscale and crop the face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_face = gray[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to 50x50
    resized_face = cv2.resize(cropped_face, (50, 50)).flatten() / 255.0

    # Perform prediction
    weight_node1 = params["weight_node1"]
    bias_node1 = params["bias_node1"]
    weight_output = params["weight_output"]
    bias_output = params["bias_output"]
    label_map = params["label_map"]

    # Forward pass through the neural network
    Zin = np.dot(resized_face, weight_node1) + bias_node1
    Zj = sigmoid(Zin)
    Yin = np.dot(Zj, weight_output) + bias_output
    Y = sigmoid(Yin)

    # Convert output to probabilities using softmax
    probabilities = softmax(Y)
    predicted_label = np.argmax(probabilities)
    confidence = probabilities[predicted_label] * 100

    print(f"Predicted: {label_map[predicted_label]}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("Class Probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  {label_map[idx]}: {prob * 100:.2f}%")

    return label_map[predicted_label], confidence

if __name__ == "__main__":
    # Replace "path/to/new_image.png" with the path to your test image
    if "--help" in sys.argv[1:]:
        print('Cara pakai:\npy predict.py "gambar.jpg atau .png atau .bmp tidak bisa .heic"')
        exit(0)
    else:
        pass
    image_path = f'{sys.argv[1]}'
    preprocess_and_predict(image_path)
