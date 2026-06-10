# Simple Artificial Neural Network (ANN) - Single Neuron (Perceptron) - OR Gate Example

from tensorflow.keras.models import Sequential # Neural Network model ko layers ki sequence (tarteeb) mein define karne ke liye.
from tensorflow.keras.layers import Dense      # Model mein layers (neurons) add karne ke liye.
import numpy as np                             # Numerical operations aur data array banane ke liye.

# 1. OR Gate Data
X = np.array([[0,0], [0,1], [1,0], [1,1]]) # Input data (A aur B) ka array banaya.
y = np.array([[0], [1], [1], [1]])         # Output data (A OR B) ka array banaya.

# 2. Model: Single Layer ka model banana
model = Sequential() # Sequential model ko initialize kiya.
# Dense layer: 1 neuron (output), 2 inputs (input_dim=2), 'sigmoid' activation use ki.
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. Compile Model
model.compile(
    optimizer='sgd',             # Optimization algorithm: Stochastic Gradient Descent (SGD).
    loss='binary_crossentropy',  # Loss function: Binary Classification ke liye cross-entropy loss.
    metrics=['accuracy']         # Metric: Accuracy ko monitor karna hai.
)

# 4. Model Train karna
model.fit(X, y, epochs=1000, verbose=0) # Data par model ko 1000 dafa train kiya. verbose=0 se training output chupaya.

# 5. Test Model
print("Predictions:") # Prediction results print karna.
print(model.predict(X).round()) # Model ki predictions nikal kar 0 ya 1 mein round off kar diya.