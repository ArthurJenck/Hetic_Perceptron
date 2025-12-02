from perceptron import Perceptron

n_iterations = 2000

perceptron = Perceptron(n_iterations=n_iterations)

X, y = perceptron.getData('iris.csv')

X_train, y_train, X_test, y_test = perceptron.train_test_split(X, y, test_size=0.2)

perceptron.fit(X_train, y_train)

train_accuracy = perceptron.score(X_train, y_train)
print(f"Précision : {train_accuracy * 100:.2f}%")

test_accuracy = perceptron.score(X_test, y_test)
print(f"Précision sur les données de test : {test_accuracy * 100:.2f}%")