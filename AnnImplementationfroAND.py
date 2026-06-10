from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. AND Gate Input Data (X) aur Output Data (y)
# Input X (A, B): [0,0], [0,1], [1,0], [1,1]
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# Output y (A AND B): [0], [0], [0], [1]
y = np.array([[0],[0],[0],[1]])

# 2. Model: Sequential model banana
model = Sequential()
# Single Layer: Ek hi neuron (output layer), kyunki AND gate bhi linear separable hai
# input_dim=2: Do inputs (A aur B)
# activation='sigmoid': Output ko 0 aur 1 ke beech compress karta hai
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. Model Compile karna
# optimizer='sgd': Simple Stochastic Gradient Descent
# loss='binary_crossentropy': Binary classification loss
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Model Train karna
# epochs=2000
model.fit(X, y, epochs=2000, verbose=0)

# 5. Accuracy Evaluate karna
loss, acc = model.evaluate(X, y, verbose=0)
print("AND Gate Model Accuracy:", acc)

# 6. Final Predictions check karna
print("AND Gate Predictions (Input -> Predicted Output):")
print(model.predict(X).round())