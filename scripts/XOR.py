import numpy as np
from sklearn.neural_network import MLPClassifier
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]) # Added closing square bracket
Y = np.array([0,1,1,0])
model = MLPClassifier(
    hidden_layer_sizes = (4),
    activation = 'tanh', # Added comma
    solver = 'lbfgs', # Corrected 'ibfgs' to 'lbfgs' and added closing quote and comma
    max_iter = 10000, # Added comma
    random_state = 42
)
model.fit(X,Y) # Changed 'y' to 'Y' for consistency
predictions = model.predict(X)
print("predictions.")
print(predictions)
print("\nActual Output:") # Corrected /n to \n
print(Y) # Changed 'y' to 'Y' for consistency
print("\nAccuracy:",model.score(X,Y)) # Corrected /n to \n and 'y' to 'Y'
