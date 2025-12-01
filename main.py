from perceptron import Perceptron

perceptron = Perceptron()

X, y = perceptron.getData('iris.csv')

perceptron.fit(X, y)

# example = X[0]
# prediction_class = perceptron.predict(example, perceptron.weights)
