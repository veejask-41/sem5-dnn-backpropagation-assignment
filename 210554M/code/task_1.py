import csv
import numpy as np


def write_W_list_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each row from the 2D list into the CSV
        for w in data:
          writer.writerows(w[0])
def write_b_list_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each row from the 2D list into the CSV
        for w in data:
          writer.writerows(w[1])

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z -max(Z))  # Stability improvement for large values
    return expZ / expZ.sum(axis=0, keepdims=True)

# Loss function (cross-entropy)
def cross_entropy_loss(A, Y):
    return -np.sum(Y * np.log(A + 1e-8))

# Neural Network class
class NeuralNetwork:
    def __init__(self):
        self.global_gradients=[]
        self.layers = []
        self.activations = []
        self.weights = []
        self.biases = []
        self.layer_outputs = []  # Cache for forward pass

    def add_layer(self, input_size, output_size, w,b, activation='relu'):
        # Initialize weights and biases
        # W = np.random.randn(input_size, output_size) * 0.01
        # b = np.zeros((output_size,))
        self.weights.append(w)
        self.biases.append(b)

        # Store the activation function for this layer
        if activation == 'relu':
            self.activations.append(relu)
        elif activation == 'softmax':
            self.activations.append(softmax)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Keep track of layers
        self.layers.append((input_size, output_size, activation))

    def forward(self, X):
        self.layer_outputs = []
        A = X

        for i, activation in enumerate(self.activations):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = activation(Z)
            self.layer_outputs.append((Z, A))

        return A

    def compute_loss(self, A, Y):
        return cross_entropy_loss(A, Y)

    def backward(self, X, Y):
      gradients = []
      m = Y.shape[0]  # Number of samples (assuming batch size is 1 in your case)

      # Backpropagate through layers
      dA = self.layer_outputs[-1][1] - Y  # Output layer error (A - Y)

      for i in reversed(range(len(self.layers))):
          Z, A = self.layer_outputs[i]
          dZ = dA

          # Compute gradients
          # Reshape dZ and the previous layer's output to ensure correct matrix multiplication
          A_prev = self.layer_outputs[i-1][1] if i > 0 else X
          dW = np.dot(A_prev.T.reshape(-1, 1), dZ.reshape(1, -1)) / m
          db = np.sum(dZ, axis=0, keepdims=True) / m

          self.global_gradients.append([dW,db])

          gradients.insert(0, [dW, db])  # Insert gradients at the beginning

          if i > 0:  # If not the input layer
              dA = np.dot(dZ, self.weights[i].T) * relu_derivative(self.layer_outputs[i-1][0])

      return gradients


    def update_parameters(self, gradients, learning_rate):
    # Update weights and biases using gradients
      for i, (dW, db) in enumerate(gradients):
          self.weights[i] -= learning_rate * dW
          self.biases[i] -= learning_rate * db.squeeze()  # Use squeeze() to match dimensions


import numpy as np

W0 = []
with open(r'Task_1\b\w-100-40-4.csv', 'r') as file:
    for line in file:
        # Split by delimiter (comma), skip the first column, and convert remaining values to float
        values = line.strip().split(',')[1:]
        W0.append([float(value) for value in values])

# Reading b0
b0 = []
with open(r'Task_1\b\b-100-40-4.csv', 'r') as file:
    for line in file:
        # Split by delimiter (comma), skip the first column, and convert remaining values to float
        values = line.strip().split(',')[1:]
        b0.append([float(value) for value in values])

W1= np.array(W0[:14])
W2= np.array(W0[14:114])
W3=np.array(W0[114:])
b1=np.array(b0[0])
b2=np.array(b0[1])
b3=np.array(b0[2])
  # Example usage
if __name__ == "__main__":
    # Initialize network
    nn = NeuralNetwork()

    # Define network structure
    nn.add_layer(input_size=14, output_size=100,w=W1,b=b1, activation='relu')
    nn.add_layer(input_size=100,w=W2,b=b2, output_size=40 ,activation='relu')
    nn.add_layer(input_size=40,w=W3,b=b3, output_size=4, activation='softmax')  # Output layer

    # Input data (X) and one-hot encoded labels (Y)
    X = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1])  # Example input data

    label = 3  # Example class label
    Y = np.zeros((1, 4))  # One-hot encoding
    Y[0, label ] = 1
    # print(Y)


    # Forward propagation
    A = nn.forward(X)
    # print("Output after forward propagation:", A)

    # Compute loss
    loss = nn.compute_loss(A, Y)
    # print(f"Loss: {loss}")

    # Backward propagation
    gradients = nn.backward(X, Y)

    # Update parameters
    nn.update_parameters(gradients, learning_rate=0.01)

    # gradients

#     print("Gradients for W1:", gradients[0][0],gradients[0][0].shape)
    nn.global_gradients.reverse()

    write_W_list_to_csv(nn.global_gradients, r'210554M\Task_1\dw.csv')
    write_b_list_to_csv(nn.global_gradients, r'210554M\Task_1\db.csv')