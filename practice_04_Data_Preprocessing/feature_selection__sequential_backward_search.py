from itertools import combinations
import numpy as np

# =============================================================================
# for loading data
# =============================================================================
from preprocess_util import Data_Fetcher, Preprocess_Util

# =============================================================================
# for estimator
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# for SBS algorithm
# =============================================================================
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================================================
# for plotting
# =============================================================================
import matplotlib.pyplot as plt

class SBS():
	def __init__(self, estimator, k_features,
			     scoring=accuracy_score,
				 test_size=0.25, random_state=1):
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.scoring = scoring
		self.test_size = test_size
		self.random_state = random_state
		
	def __cal_score(self, X_train, X_test, Y_train, Y_test, indices):
		"""
		Use "partial features" with indices amid the parameter: `indices`
		to fit a common estimator temporarily,
		and use the estimator to evaluate a "score" of Y's prediction.
		> The value of "score" (e.g., accuracy measure)
		> can be used to select appropriate features
		> so as to approach the result of "feature selection" (=> `fit()`)
		"""
		# Train an estimator
		self.estimator.fit(X_train[:, indices], Y_train)
		
		# Use the estimator & test features to predict test labels (Y_test_hat)
		Y_predicted = self.estimator.predict(X_test[:, indices])
		
		# Evaluate the "score" between realistic test labels (Y_test_hat)
		# and predicted test labels (Y_test_predicted)
		score = self.scoring(Y_test, Y_predicted)
		return score
		
	#def fit(self, X_train, X_test, Y_train, Y_test):
	def fit(self, X_train, Y_train):
		X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=self.test_size, random_state=self.random_state)
		n_dims = X_train.shape[1] # Suppose n_dims is N, remain k feature(s)
		best_indices = tuple(range(n_dims)) # initial dimensionality: N
		# ---
		self.raw_feature_num = n_dims
		self.best_indices_list = [best_indices] # for histotical plotting
		self.best_score_list = list() # for histotical plotting
		# ---
		init_score = self.__cal_score(X_train, X_test, Y_train, Y_test, best_indices) 
		self.best_score_list.append(init_score) # [0.9333333333333333]
		
		"""
		Time complexity
		---
		T(N) = O(N * (N-k)) | N: features' dimensionality of raw dataset
		---
		> If N >> k, T(N) ~= O(N^2)
		> Otherwise, if (N-k) ~= 1, T(N) ~= O(N)
		"""

		while n_dims > self.k_features: # need (N-k) rounds
			best_score_ = -np.inf
			
			''' List indices for all possible "feature subsets" with (N-1) dim. '''
			for tmp_indices in combinations(best_indices, n_dims-1): # need N rounds
				tmp_score = self.__cal_score(X_train, X_test, Y_train, Y_test, tmp_indices)
				if tmp_score > best_score_:
					best_indices = tmp_indices
					best_score_ = tmp_score
			
			self.best_indices_list.append(best_indices)
			self.best_score_list.append(best_score_)
			n_dims -= 1
	
	def plot_history(self):
		num_features_list = [len(e) for e in self.best_indices_list]
		plt.plot(num_features_list, self.best_score_list, marker='o')
		plt.ylim([0.7, 1.02])
		plt.ylabel("Accuracy")
		plt.xlabel("Number of features")
		plt.grid()
		plt.tight_layout()
		plt.show()
		
	def get_reduced_feature_indices(self, df, statisfied_feature_num):
		indices = list(self.best_indices_list[self.raw_feature_num - statisfied_feature_num])
		statisfied_features = df.columns[1:][indices]
		print(f"reduced features: {statisfied_features}\n")
		return indices
		
if __name__ == "__main__":
	preprocess_util = Preprocess_Util()
	data_fetcher = Data_Fetcher()
	df = data_fetcher.fetch_data("wine-data-01")
	
	#test_dataset_ratio = 0.3
	test_dataset_ratio = 0.25
	label_col_name = "Class label"
	X, Y = preprocess_util.label_features_split(df, label_col_name)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1)
	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	regularize_type = "standardize"
	X_train, X_test = preprocess_util.regularization(X_train, X_test, regularize_type)
	
	n_neighbors = 5
	k_features = 1
	#---
	knn = KNeighborsClassifier(n_neighbors)
	sbs = SBS(knn, k_features)
	#sbs.fit(X_train, X_test, Y_train, Y_test)
	sbs.fit(X_train, Y_train)
	
	# [TEST]
	#print(sbs.best_score_list)
	'''
	[0.9333333333333333,
	 0.9777777777777777,
	 0.9777777777777777,
	 0.9777777777777777,
	 1.0,
	 1.0,
	 1.0,
	 1.0,
	 0.9777777777777777,
	 0.9555555555555556,
	 0.9555555555555556,
	 0.9333333333333333,
	 0.8]
	'''
	#print(sbs.best_indices_list)
	'''
	[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
	 (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12),
	 (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12),
	 (0, 1, 2, 3, 5, 6, 7, 9, 10, 12),
	 (0, 1, 2, 3, 6, 7, 9, 10, 12),
	 (0, 1, 2, 3, 6, 7, 9, 12),
	 (0, 1, 2, 3, 6, 9, 12),
	 (0, 1, 2, 6, 9, 12),
	 (0, 2, 6, 9, 12),
	 (0, 2, 6, 9),
	 (2, 6, 9),
	 (6, 9),
	 (6,)]
	'''
	
	sbs.plot_history()
	statisfied_feature_num = 3
	reduced_feat_indices = sbs.get_reduced_feature_indices(df, statisfied_feature_num)
	
	# ---
	knn.fit(X_train, Y_train)
	print("[INFO] Before features selection:")
	print(f"Training accuracy:\n {knn.score(X_train, Y_train)}")
	# Training accuracy: 0.9849624060150376
	print(f"Test accuracy:\n {knn.score(X_test, Y_test)}\n")
	# Test accuracy: 0.9777777777777777
	
	# ---
	X_train_reduced = X_train[:, reduced_feat_indices]
	X_test_reduced = X_test[:, reduced_feat_indices]
	knn.fit(X_train_reduced, Y_train)
	print("[INFO] After features selection:")
	print(f"Training accuracy:\n {knn.score(X_train_reduced, Y_train)}")
	# Training accuracy: 0.9849624060150376
	print(f"Test accuracy:\n {knn.score(X_test_reduced, Y_test)}\n")
	# Test accuracy: 0.9777777777777777
