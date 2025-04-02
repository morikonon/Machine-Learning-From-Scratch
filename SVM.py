import numpy as np

class SVM:
	def __init__(self , learning_rate = 0.001 , lambda_param = 0.01 , n_iters = 1000 , kernel = "linear"):
		self.learning_rate = learning_rate
		self.lambda_param = lambda_param
		self.n_iters = n_iters
		self.kernel = kernel
	
	def fit(self , X , y):
		n_samples , n_features = X.shape

		self.w = np.zeros(n_features)
		self.b = 0

		for _ in range(self.n_iters):
			for idx_i , x_i in enumerate(X):
				condition = y[idx_i] * (np.dot(x_i , self.w) + self.b) >= 1
				if condition:
					self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
				else:
					self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i , y[idx_i]) )
					self.b -= self.learning_rate * y[idx_i]

	def prediction(self , X):
		return np.sign(np.dot(X , self.w) + self.b)