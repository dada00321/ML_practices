# =============================================================================
# for dataset
# =============================================================================
from sklearn import datasets
from sklearn.model_selection import train_test_split

# =============================================================================
# for Random Forest model
# =============================================================================
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# for Plotting
# =============================================================================
from module.plotting import Plotting

# =============================================================================
# others 
# =============================================================================
import numpy as np

class Random_Forest_Builder():
	def load_iris_data(self):
		iris_data = datasets.load_iris()
		X = iris_data.data[:, [2, 3]]
		Y = iris_data.target
		return X, Y
	
	def train(self, X_train, Y_train):
		forest = RandomForestClassifier(criterion="gini",
									    n_estimators=25,
									    random_state=1,
										n_jobs=2)
		forest.fit(X_train, Y_train)
		return forest
	
	def show_random_forest(self, random_forest, classifier_name, label_names,
						   scatter_name_dict, save_path,
						   sample_size, test_dataset_ratio,
						   X_train, X_test, Y_train, Y_test):
		# plot classification result
		classifier = random_forest
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
	random_forest_builder = Random_Forest_Builder()
	X, Y = random_forest_builder.load_iris_data()
	sample_size = np.bincount(Y).sum()
	test_dataset_ratio = 0.3
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	random_forest = random_forest_builder.train(X_train, Y_train)
	
	classifier_name = "Random Forest"
	label_names = ["petal length (cm)", "petal width (cm)"]
	scatter_name_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
	save_path = "res/random_forest/sklearn_random_forest/"+\
		        "iris_random_forest__decision_boundaries.png"
	random_forest_builder.show_random_forest(random_forest, classifier_name, label_names,
						   scatter_name_dict, save_path,
						   sample_size, test_dataset_ratio,
						   X_train, X_test, Y_train, Y_test)
	