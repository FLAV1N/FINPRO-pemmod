import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# Helper functions
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def derivativeSigmoid(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

# Dataset loading and preprocessing
def load_dataset(dataset_dir):
    data = []
    labels = []
    label_map = {0: "ALVAN", 1: "ANCAS", 2: "ANGEL", 3: "ARVEL", 4: "ATHAYA", 5: "BUDI", 6: "FAIZ", 7: "HIZRI", 8: "ILHAM", 9:"IQBAL", 10:"IRFAN", 11:"JOKO", 12:"MALIK", 13:"SEKAR", 14:"TIAN"}

    for idx, person_name in enumerate(os.listdir(dataset_dir)):
        label_map[idx] = person_name
        person_path = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                file_path = os.path.join(person_path, filename)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = image.flatten() / 255.0  # Flatten and normalize
                data.append(image)
                labels.append(idx)

    return np.array(data), np.array(labels), label_map


# Neural network class
class BPNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0005):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weight_node1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_node1 = np.zeros(hidden_size)
        self.weight_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)
        self.loss_history = []  # Track loss history for plotting

    def forward(self, X):
        self.Zin = np.dot(X, self.weight_node1) + self.bias_node1
        self.Zj = sigmoid(self.Zin)
        self.Yin = np.dot(self.Zj, self.weight_output) + self.bias_output
        self.Y = sigmoid(self.Yin)
        return self.Y

    def backward(self, X, y, output, lambda_reg=0.0002):
        sigma_output = (y - output) * derivativeSigmoid(output)
        sigma_hidden = np.dot(sigma_output, self.weight_output.T) * derivativeSigmoid(self.Zj)

        # Update weights and biases with L2 regularization
        self.weight_output += self.learning_rate * (np.dot(self.Zj.T, sigma_output) - lambda_reg * self.weight_output)
        self.bias_output += self.learning_rate * np.sum(sigma_output, axis=0)
        self.weight_node1 += self.learning_rate * (np.dot(X.T, sigma_hidden) - lambda_reg * self.weight_node1)
        self.bias_node1 += self.learning_rate * np.sum(sigma_hidden, axis=0)

    def train(self, X, y, epochs=5000, patience=50, lambda_reg=0.0002):
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)  # Track loss over time

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with loss {best_loss:.4f}")
                break

            self.backward(X, y, output, lambda_reg)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.8f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Main script for training
if __name__ == "__main__":
    dataset_dir = "processed_images"
    X, y, label_map = load_dataset(dataset_dir)

    num_classes = len(label_map)
    encoder = OneHotEncoder(sparse_output=False, categories="auto")
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    hidden_size = 1024
    output_size = num_classes
    learning_rate = 0.0005

    model = BPNN(input_size, hidden_size, output_size, learning_rate)
    model.train(X_train, y_train, epochs=5000)

    # Evaluate on test set
    predictions = model.predict(X_test)
    accuracy = np.mean(np.argmax(y_test, axis=1) == predictions)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Save the model
    with open("bpnn_model.pkl", "wb") as f:
        pickle.dump({
            "weight_node1": model.weight_node1,
            "bias_node1": model.bias_node1,
            "weight_output": model.weight_output,
            "bias_output": model.bias_output,
            "label_map": label_map
        }, f)
    print("Model saved.")

    # Plot the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_history, label="Training Loss")
    plt.title("Loss Graph During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (Loss)")
    plt.legend()
    plt.grid(True)
    plt.show()
