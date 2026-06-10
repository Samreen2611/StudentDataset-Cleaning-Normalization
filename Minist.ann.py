# Multi-Layer Perceptron (MLP) Neural Network for MNIST Digit Classification

# ----------------------------
# 1. IMPORT LIBRARIES
# ----------------------------
import numpy as np                   # Numerical operations.
from keras.datasets import mnist     # MNIST handwritten digits dataset load karne ke liye.
from keras.models import Sequential  # Sequential NN model banane ke liye.
from keras.layers import Dense       # Layers add karne ke liye.
from keras.utils import to_categorical # Integer labels ko one-hot encoded vectors mein badalne ke liye.

# ----------------------------
# 2. DATA LOAD KARNA
# ----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST data ko training aur testing sets mein load kiya.

# Training ko fast karne ke liye chota subset lena (Optional, sirf demo ke liye)
x_train = x_train[:10000] # Training images ka size 10000 tak reduce kiya.
y_train = y_train[:10000] # Training labels ka size 10000 tak reduce kiya.

x_test = x_test[:2000] # Testing images ka size 2000 tak reduce kiya.
y_test = y_test[:2000] # Testing labels ka size 2000 tak reduce kiya.

# ----------------------------
# 3. PREPROCESSING
# ----------------------------
# Image ko 2D (28x28) se 1D vector (784) mein badalna aur 0-1 tak normalize karna.
x_train = x_train.reshape(x_train.shape[0], 28*28).astype("float32") / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype("float32") / 255

# Target labels ko One-hot encode karna (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
y_train = to_categorical(y_train, 10) # Training labels ko one-hot encoded kiya (10 classes).
y_test = to_categorical(y_test, 10) # Testing labels ko one-hot encoded kiya.

# ----------------------------
# 4. MODEL BANANA (MLP with 2 hidden layers)
# ----------------------------
model = Sequential() # Sequential model initialize kiya.
# Hidden Layer 1: 64 neurons, 'relu' activation, 784 inputs (Input Layer).
model.add(Dense(64, activation='relu', input_shape=(784,)))
# Hidden Layer 2: 32 neurons, 'relu' activation.
model.add(Dense(32, activation='relu'))
# Output Layer: 10 neurons (0-9 digits), 'softmax' activation (probability distribution ke liye).
model.add(Dense(10, activation='softmax'))

# ----------------------------
# 5. MODEL COMPILE KARNA
# ----------------------------
model.compile(
    optimizer='adam',                # Optimization algorithm: Adam (bahut effective).
    loss='categorical_crossentropy', # Loss function: Multi-class classification (One-hot encoding ke saath) ke liye.
    metrics=['accuracy']             # Metric: Accuracy.
)

# ----------------------------
# 6. MODEL TRAIN KARNA
# ----------------------------
model.fit(
    x_train, y_train,
    batch_size=64,      # Har step mein 64 samples ko process karega.
    epochs=2,           # Model sirf 2 baar poore data se seekhega (demo ke liye kam rakha).
    validation_split=0.1 # 10% training data ko validation ke liye alag rakha.
)

# ----------------------------
# 7. MODEL EVALUATE KARNA
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test) # Test data par model ki performance check ki.
print("\nTest Loss:", test_loss) # Test loss print kiya.
print("Test Accuracy:", test_acc) # Test accuracy print ki.