import csv
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}


    def getData(self, filename):
        X_data = []
        y_data = []

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter= ",")

            # Passer les en-têtes
            next(reader)

            for row in reader:
                features = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                X_data.append(features)
                
                y_data.append(row[4])

        X = np.array(X_data)
        y = np.array(y_data)

        return X, y

    def train_test(self, X, y, test_size=0.2):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        split_idx = int(len(X) * (1 - test_size))

        X_train = X_shuffled[:split_idx]
        y_train = y_shuffled[:split_idx]
        X_test = X_shuffled[split_idx:]
        y_test = y_shuffled[split_idx:]

        return X_train, y_train, X_test, y_test 

    def scalar_product(self, a, b):
        return np.dot(a, b)

    def activation_function(self, value):
        return 1 if value > 0 else 0

    def predict(self, example, weights):
        dot = self.scalar_product(example, weights)
        return self.activation_function(dot)

    def fit(self, X, y):
        learning_rate = self.learning_rate
        n_iterations = self.n_iterations
        label_map = self.label_map

        y_encoded=[]
        for label in y:
            y_encoded.append(label_map[label])

        y_encoded = np.array(y_encoded)

        n_classes = len(label_map)
        
        self.weights = np.random.randn(n_classes, 4)

        for epoch in range(n_iterations):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y_encoded[indices]
            
            for neuron in range(n_classes):
                for i in range(len(X_shuffled)):
                    example = X_shuffled[i]
                    true_class = y_shuffled[i]

                    true_response = 1 if true_class == neuron else 0

                    prediction = self.predict(example, self.weights[neuron])

                    error = true_response - prediction

                    self.weights[neuron] += learning_rate * error * example

    def predict_class(self, example):
        scores = []
        for neuron in range(3):
            score = self.scalar_product(example, self.weights[neuron])
            scores.append(score)

        return np.argmax(scores)

    def score(self, X, y):
        label_map = self.label_map
        y_encoded = np.array([label_map[label] for label in y])

        correct = 0
        for i in range(len(X)):
            prediction = self.predict_class(X[i])
            if prediction == y_encoded[i]:
                correct += 1

        accuracy = correct / len(X)
        return accuracy