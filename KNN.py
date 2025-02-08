import numpy as np
from collections import Counter

def euclidean_distance(x1 , x2):
	distances = np.sqrt(np.sum((x1 - x2) ** 2))
	return distances

class KNN:
	def __init__(self , k = 3):
		self.k = k

	def fit(self , X , y):
		self.X = X
		self.y = y

	def predict(self , X):
		predictions = [self._predict(x) for x in X]
		return predictions
	def _predict(self , X):
		distances = [euclidean_distance(X , x_train) for x_train in self.X]

		k_indeces = np.argsort(distances)[:self.k]
		k_labels = [self.y[i] for i in k_indeces]

		most_common = Counter(k_labels).most_common()
		return most_common[0][0]



		