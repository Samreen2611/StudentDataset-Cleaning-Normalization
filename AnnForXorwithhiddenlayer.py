from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. XOR Gate Input Data (X) aur Output Data (y)
# Input X (A, B): [0,0], [0,1], [1,0], [1,1]
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# Output y (A XOR B): [0], [1], [1], [0] (Different inputs = 1, Same inputs = 0)
y = np.array([[0],[1],[1],[0]])

# 2. Model: Sequential model banana (Multi-Layer Perceptron - MLP)
model = Sequential()

# Hidden layer: 2 neurons
# input_dim=2: Input layer ke liye
# activation='sigmoid': Nonlinearity add karne ke liye zaroori, jo XOR ko solve karti hai
model.add(Dense(2, input_dim=2, activation='sigmoid'))

# Output layer: 1 neuron
# activation='sigmoid': Final binary output (0 ya 1) ke liye
model.add(Dense(1, activation='sigmoid'))

# 3. Model Compile karna
# optimizer='adam': SGD se behtar performance ke liye Adam use kiya gaya hai
# loss='binary_crossentropy'
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Model Train karna
# epochs=5000: XOR mushkil hai, isliye zyada epochs (training cycles) zaroori hain
model.fit(X, y, epochs=5000, verbose=0)

# 5. Accuracy Evaluate karna
loss, acc = model.evaluate(X, y, verbose=0)
print("XOR Gate Model Accuracy:", acc)

# 6. Final Predictions check karna
print("XOR Gate Predictions (Input -> Predicted Output):")
print(model.predict(X).round())