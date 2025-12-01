import numpy as np
from perceptron import Perceptron

perceptron = Perceptron()

X, y = perceptron.getData('iris.csv')

example = X[0]
weights = np.random.randn(4)

prediction_class = perceptron.predict(example, weights)

print(prediction_class)

# perceptron.fit(X, y)