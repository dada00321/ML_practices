import numpy as np

# =============================================================================
# for dataset
# =============================================================================
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# =============================================================================
# for KNN model
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# for Plotting
# =============================================================================
from module.plotting import Plotting

class KNN_Builder():
	def normalize_input_features(self, X_train, X_test):
		'''
		rtn_val: normalized X_train and X_test (via same statistics)
		'''
		ss = StandardScaler()
		ss.fit(X_train) # calculate statistics for normalization
		return ss.transform(X_train), ss.transform(X_test)

	def load_iris_data(self):
		iris_data = datasets.load_iris()
		X = iris_data.data[:, [2, 3]]
		Y = iris_data.target
		return X, Y
	
	def train(self, X_train, Y_train):
		knn_model = KNeighborsClassifier(n_neighbors=5,
									     p=2,
									     metric="minkowski")
		knn_model.fit(X_train, Y_train)
		return knn_model
	
	def show_random_forest(self, knn_model, classifier_name, label_names,
						   scatter_name_dict, save_path,
						   sample_size, test_dataset_ratio,
						   X_train, X_test, Y_train, Y_test):
		# plot classification result
		classifier = knn_model
		x_label = label_names[0]
		y_label = label_names[1]
		plotting = Plotting(classifier,
						    classifier_name,
						    x_label,
							y_label,
							scatter_name_dict)
		X_combined = np.vstack((X_train, X_test))
		Y_combined = np.hstack((Y_train, Y_test))
		test_idx = range(int(sample_size*(1-test_dataset_ratio)), sample_size)
		plotting.plot_classification(X_combined, Y_combined,
								     save_path, test_idx)

if __name__ == "__main__":
	knn_forest_builder = KNN_Builder()
	X, Y = knn_forest_builder.load_iris_data()
	sample_size = np.bincount(Y).sum()
	test_dataset_ratio = 0.3
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	X_train, X_test = knn_forest_builder.normalize_input_features(X_train, X_test)
	knn_model = knn_forest_builder.train(X_train, Y_train)
	
	classifier_name = "Random Forest"
	label_names = ["petal length (cm)", "petal width (cm)"]
	scatter_name_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
	save_path = "res/KNN/sklearn_KNN/"+\
		        "iris_KNN__decision_boundaries.png"
	knn_forest_builder.show_random_forest(knn_model, classifier_name, label_names,
						   scatter_name_dict, save_path,
						   sample_size, test_dataset_ratio,
						   X_train, X_test, Y_train, Y_test)
