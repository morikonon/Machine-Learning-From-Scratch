import numpy as np

class LinearRegression:
	def __init__(self , n_iter = 1000 , learning_rate = 0.001):
		self.n_iter = n_iter
		self.learning_rate = learning_rate
		self.weights = None
		self.bias = None
	def fit(self , X , y):
		n_samples , n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iter):
			predict = np.dot(X , self.weights) + self.bias

			error = y - predict
			dw = (-2 / n_samples ) * np.dot(X.T , error)
			db = (-2 / n_samples) * np.sum(error)

			self.weights = self.weights - self.learning_rate * dw
			self.bias = self.bias - self.learning_rate * db
	def predict(self, X):
		predict = np.dot(X , self.weights) + self.bias
		return predict