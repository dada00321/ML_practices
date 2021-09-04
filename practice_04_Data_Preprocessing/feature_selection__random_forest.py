import numpy as np

# =============================================================================
# for feature selection --- random forest
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# =============================================================================
# for loading data
# =============================================================================
from preprocess_util import Data_Fetcher, Preprocess_Util

# =============================================================================
# for plotting
# =============================================================================
import matplotlib.pyplot as plt

def list_feature_importances(X_train, Y_train):
	forest = RandomForestClassifier(n_estimators=500, random_state=1)
	forest.fit(X_train, Y_train)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	for i, feature_idx in enumerate(indices):
		print(f"{i+1:2d}) {feature_names[feature_idx]:28s} {importances[feature_idx]}")
	print()
	return forest, importances, indices

def plot_feature_importances(X_train, importances, feature_names, indices):
	plt.title("Feature Importance")
	plt.bar(range(X_train.shape[1]),
		    importances[indices],
			align="center")
	plt.xticks(range(X_train.shape[1]),
			   feature_names[indices],
			   rotation=90)
	plt.xlim([-1, X_train.shape[1]])
	plt.tight_layout()
	plt.savefig("./res/feature_selection__decision_tree"+\
			    "/feature_importance.png")
	plt.show()

if __name__ == "__main__":
	dataset_name = "wine-data-01"
	label_col_name = "Class label"
	test_dataset_ratio = 0.3
	random_state = 1
	
	# 1. Load data
	preprocess_util = Preprocess_Util()
	data_fetcher = Data_Fetcher()
	df = data_fetcher.fetch_data(dataset_name)
	feature_names = df.columns[1:]
	
	# 2. Preprocess data
	X, Y = preprocess_util.label_features_split(df, label_col_name)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=random_state)
	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=random_state, stratify=Y)
	list_feature_importances(X_train, Y_train)
	
	# 3. List feature from most-importance to last-importance to the labels (Y)
	# => Importances calculate by `RandomForestClassifier`
	forest, importances, indices = list_feature_importances(X_train, Y_train)
	
	# 4. Plot result
	plot_feature_importances(X_train, importances, feature_names, indices)
	
	# 5. Feature selection
	sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
	X_selected = sfm.transform(X_train)
	print(f"Number of features: {X_selected.shape[1]}")
	for i in range(X_selected.shape[1]):
		feature_idx = indices[i]
		print(f"{i+1:2d}) {feature_names[feature_idx]:28s} {importances[feature_idx]}")
