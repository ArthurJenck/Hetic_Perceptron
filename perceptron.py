import csv
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        return None

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

    def scalar_product(self, a, b):
        return np.dot(a, b)

    def activation_function(self, value):
        return 1 if value > 0 else 0

    def predict(self, example, weights):
        dot = self.scalar_product(example, weights)
        return self.activation_function(dot)