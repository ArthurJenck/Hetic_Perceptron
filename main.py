from perceptron import Perceptron

n_iterations = 2000

perceptron = Perceptron(n_iterations=n_iterations)

X, y = perceptron.getData('iris.csv')

perceptron.fit(X, y)

for i in range(10):
    example = X[i]
    prediction = perceptron.predict_class(example)

accuracy = perceptron.score(X, y)
print(f"Precision: {accuracy * 100:.2f}%")