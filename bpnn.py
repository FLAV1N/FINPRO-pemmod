import numpy as np

# Helper functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivativeSigmoid(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

np.set_printoptions(precision=4, suppress=True)

# Data
x_actual = np.array([
    [0.85, 0.15, 0.15],
    [0.82, 0.12, 0.12],
    [0.14, 0.84, 0.14],
    [0.12, 0.82, 0.12],
    [0.15, 0.15, 0.85],
    [0.13, 0.13, 0.83]
])

y_actual = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])

# Initial weights and biases
weight_node1 = np.array([0.25, 0.25, 0.25])
bias_node1 = 0.3

weight_node2 = np.array([0.25, 0.25, 0.25])
bias_node2 = 0.3

weight_output1 = np.array([0.35, 0.25])
bias_output1 = 0.3

weight_output2 = np.array([0.35, 0.25])
bias_output2 = 0.3

weight_output3 = np.array([0.35, 0.25])
bias_output3 = 0.3

LR = 0.25

# Training
for j in range(3):  # Minimum 3 iterations
    total_error = 0
    print(f"Epoch {j + 1}")
    for i, (x, y) in enumerate(zip(x_actual, y_actual)):
        """
        FORWARD
        """
        # Node 1
        Zin1 = bias_node1 + weight_node1.dot(x)
        Zj1 = sigmoid(Zin1)

        # Node 2
        Zin2 = bias_node2 + weight_node2.dot(x)
        Zj2 = sigmoid(Zin2)

        # Combine hidden nodes
        Z = np.array([Zj1, Zj2])

        # Outputs
        Yin1 = bias_output1 + weight_output1.dot(Z)
        Y1 = sigmoid(Yin1)

        Yin2 = bias_output2 + weight_output2.dot(Z)
        Y2 = sigmoid(Yin2)

        Yin3 = bias_output3 + weight_output3.dot(Z)
        Y3 = sigmoid(Yin3)

        Y = np.array([Y1, Y2, Y3])

        # Error for the current data point
        e_p = 0.5 * np.sum((y - Y) ** 2) / len(Y)
        total_error += e_p

        """
        BACKWARD
        """
        # Output layer errors
        sigma1 = (y[0] - Y1) * derivativeSigmoid(Yin1)
        sigma2 = (y[1] - Y2) * derivativeSigmoid(Yin2)
        sigma3 = (y[2] - Y3) * derivativeSigmoid(Yin3)

        # Output layer weight updates
        del_weight_output1 = LR * sigma1 * Z
        del_bias_output1 = LR * sigma1

        del_weight_output2 = LR * sigma2 * Z
        del_bias_output2 = LR * sigma2

        del_weight_output3 = LR * sigma3 * Z
        del_bias_output3 = LR * sigma3

        # Hidden layer errors
        sigma_in1 = (sigma1 * weight_output1[0]) + (sigma2 * weight_output2[0]) + (sigma3 * weight_output3[0])
        sigma_in2 = (sigma1 * weight_output1[1]) + (sigma2 * weight_output2[1]) + (sigma3 * weight_output3[1])

        sigma_n1 = sigma_in1 * derivativeSigmoid(Zin1)
        sigma_n2 = sigma_in2 * derivativeSigmoid(Zin2)

        # Hidden layer weight updates
        del_weight_node1 = LR * sigma_n1 * x
        del_bias_node1 = LR * sigma_n1

        del_weight_node2 = LR * sigma_n2 * x
        del_bias_node2 = LR * sigma_n2

        # Update weights and biases
        weight_output1 += del_weight_output1
        bias_output1 += del_bias_output1

        weight_output2 += del_weight_output2
        bias_output2 += del_bias_output2

        weight_output3 += del_weight_output3
        bias_output3 += del_bias_output3

        weight_node1 += del_weight_node1
        bias_node1 += del_bias_node1

        weight_node2 += del_weight_node2
        bias_node2 += del_bias_node2

        print(f"Data {i + 1}: Error e_p = {e_p:.4f}")

    print(f"Total Error for Epoch {j + 1}: E_P = {total_error:.4f}")
    if total_error < 0.01:
        print("Stopping condition reached.")
        break
