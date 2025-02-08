import numpy as np

def sigmoid(z):
	return (1/(1 + np.exp(-z)))

class LogisticRegression:
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
			linear_predict = np.dot(X,self.weights) + self.bias
			predict = sigmoid(linear_predict)

			error = predict - y

			dw = (1/n_samples)*np.dot(X.T,error)
			db = (1/n_samples)*np.sum(error)

			self.weights = self.weights - self.learning_rate * dw
			self.bias = self.bias - self.learning_rate * db
	
	def predict(self , X):
		linear_predict = np.dot(X , self.weights) + self.bias
		predict = sigmoid(linear_predict)
		class_pred = [0 if y < 0.5 else 1 for y in predict]
		return class_pred