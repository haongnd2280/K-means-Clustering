import numpy as np
from scipy.spatial.distance import cdist  # Euclide distance by default
from tqdm import tqdm


class KMeans:

	def __init__(self, K):
		self.K = K                            # number of clusters

	def _init_kmeans(self, X):
		"""Randomly pick k rows of X as initial centers.
		"""

		self.X = X  # data matrix
		self.N = X.shape[0]                   # number of data points
		self.D = X.shape[1]                   # number of dimensions

		idx = np.random.choice(self.N, self.K, replace=False)  # choose K random indices in [0, N]
		self.centers = [self.X[idx]]          # centers matrix
		self.labels = []                      # labels matrix

	def _assign_labels(self, X, centers):
		"""Calculate pairwise distances between each data points and centers.
		"""

		D = cdist(X, centers)                 # N x K matrix
		new_labels = np.argmin(D, axis=1)  # return index of the closest center of each data points

		return new_labels

	def _update_centers(self, X, labels):
		"""Update new centers based on the labels.
		"""

		new_centers = np.zeros((self.K, self.D))
		for k in range(self.K):
			X_k = X[labels == k, :]  # collect all points assigned to the k-th cluster

			new_centers[k, :] = np.mean(X_k, axis=0)

		return new_centers

	def _check_convergence(self, centers, new_centers):
		"""Check if the algorithm has convergent or not yet.

		Return True if two sets of centers are the same.
		"""

		return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

	def fit(self, X, max_iter=100):
		"""Fit K-means clustering algorithm.
		"""

		self._init_kmeans(X)  # create initial centers

		for iter in tqdm(range(max_iter)):
			new_labels = self._assign_labels(X, self.centers[-1])
			self.labels.append(new_labels)

			new_centers = self._update_centers(X, new_labels)
			if self._check_convergence(new_centers, self.centers[-1]):
				break

			self.centers.append(new_centers)

		print("The algorithm terminates at {}th iteration.".format(iter))
