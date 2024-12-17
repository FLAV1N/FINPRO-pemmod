import sys
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from bpnn import sigmoid  # Import sigmoid function from the training code

resolusi = 50 #Pixel

# Load the DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Path to the prototxt file
    "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file
)



# Softmax function with temperature scaling

def softmax(logits, temperature=0.2):  
    exp_logits = np.exp((logits - np.max(logits)) / temperature)
    return exp_logits / np.sum(exp_logits)



# DNN-based face detection

def detect_faces_dnn(image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    h, w = image.shape[:2]



    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces



# Preprocess and predict function

def preprocess_and_predict(image_path, model_path="bpnn_model.pkl"):
    # Load the model
    with open(model_path, "rb") as f:
        params = pickle.load(f)

    # Preprocess the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return None

    faces = detect_faces_dnn(image)
    if len(faces) == 0:
        print(f"No face detected in {image_path}.")
        return None

    x, y, w, h = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = max(w, h)
    half_size = crop_size // 2

    # Crop face region

    crop_x1 = max(center_x - half_size, 0)
    crop_y1 = max(center_y - half_size, 0)
    crop_x2 = min(crop_x1 + crop_size, image.shape[1])
    crop_y2 = min(crop_y1 + crop_size, image.shape[0])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_face = gray[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_face = cv2.resize(cropped_face, (resolusi, resolusi)).flatten() / 255.0

    # Neural network forward pass
    weight_node1 = params["weight_node1"]
    bias_node1 = params["bias_node1"]
    weight_output = params["weight_output"]
    bias_output = params["bias_output"]
    label_map = params["label_map"]

    Zin = np.dot(resized_face, weight_node1) + bias_node1
    Zj = sigmoid(Zin)
    Yin = np.dot(Zj, weight_output) + bias_output
    Y = sigmoid(Yin)



    # Apply softmax for probabilities
    probabilities = softmax(Y)
    predicted_label = np.argmax(probabilities)
    confidence = probabilities[predicted_label] * 100

    print(f"Predicted: {label_map[predicted_label]}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("Class Probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  {label_map[idx]}: {prob * 100:.2f}%")
    return predicted_label, label_map



# Evaluate predictions and plot confusion matrix

def evaluate_predictions(image_dir, model_path="bpnn_model.pkl"):
    y_true = []
    y_pred = []



    # Load all test images from the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            true_label = filename.split('_')[0].upper()
            image_path = os.path.join(image_dir, filename)
            predicted_label, label_map = preprocess_and_predict(image_path, model_path)
            if predicted_label is not None:
                y_true.append(true_label)
                y_pred.append(label_map[predicted_label])



    # Generate confusion matrix
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    if "--help" in sys.argv[1:]:
        print('Usage:\npy predict.py "image_directory"')
        sys.exit(0)

    image_dir = sys.argv[1]
    evaluate_predictions(image_dir)
