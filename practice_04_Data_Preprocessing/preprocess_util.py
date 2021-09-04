import pandas as pd
import numpy as np
from io import StringIO
from time import time

""" Pre-process data """
# =============================================================================
# for padding missing values
# =============================================================================
from sklearn.impute import SimpleImputer

# =============================================================================
# for encoding nomial features
# =============================================================================
from sklearn.preprocessing import OneHotEncoder 
#from sklearn.compose import ColumnTransformer

# =============================================================================
# for spliting data into training data & test data
# =============================================================================
from sklearn.model_selection import train_test_split

# =============================================================================
# Regularize features
# =============================================================================
from sklearn.preprocessing import StandardScaler

""" Pre-process model """
from sklearn.linear_model import LogisticRegression

class Data_Fetcher():
	def fetch_data(self, data_name):
		if data_name == "example-values-01":
			csv_data = \
			"""
			A, B, C, D
			1.0,2.0,3.0,4.0
			5.0,6.0,, 8.0
			10.0,11.0,12.0,
			""".replace('\t', '')
			df = pd.read_csv(StringIO(csv_data))
			return df
		
		elif data_name == "apparel-data-01":
			apparel_data = [["green", 'M', 10.1, "T-shirt"],
							["red", 'L', 13.5, "T-shirt"],
							["blue", "XL", 15.3, "jacket"]]
			column_names = ["color", "size", "price", "class_label"]
			df = pd.DataFrame(apparel_data)
			df.columns = column_names
			return df
		
		elif data_name == "wine-data-01":
			'''
			178 rows (records): header is not included
			14 cols: 1 label + 13 features
			'''
			csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
			df = pd.read_csv(csv_url, header=None)
			#print(df.shape) # (178, 14)
			feature_names = \
			["Alcohol",
			 "Malic acid",
			 "Ash",
			 "Alcalinity of ash",
			 "Magnesium",
			 "Total phenols",
			 "Flavanoids",
			 "Nonflavanoid phenols",
			 "Proanthocyanins",
			 "Color intensity",
			 "Hue",
			 "OD280/OD315 of diluted wines",
			 "Proline"]
			df.columns = ["Class label"] + feature_names
			#print(df.shape) # (178, 14)
			return df
		
		else:
			print("[WARNING] Cannot execute fetch_data()"+\
		          "because value of `data_name` is invalid.")

class Preprocess_Util():
	""" Pre-process data """
	'''
	  Pre-process missing values
	'''
	def del_missing_values(self, data, del_direction):
		if type(data).__name__ == "DataFrame":
			if del_direction in ("row", "col"):
				if del_direction == "row":
					return data.dropna(axis=0)
				else:
					return data.dropna(axis=1)
			else:
				print("[WARNING] Cannot execute del_missing_values()"+\
		              "because value of `del_direction` is invalid.")
				return None
		else:
			print("[WARNING] Cannot execute del_missing_values()"+\
		          "because data-type of `data` is unknown.")
			return None
	
	def padding_missing_values(self, data, method):
		if type(data).__name__ == "DataFrame":
			if method in ("mean-pandas", "mean-sklearn"):
				if method == "mean-pandas":
					return data.fillna(data.mean())
				else:
					cols = data.columns
					imr = SimpleImputer(missing_values=np.nan, strategy="mean")
					imr = imr.fit(data.values)
					imputed_data = pd.DataFrame(imr.transform(data.values), columns=cols)
					return imputed_data
			else:
				print("[WARNING] Cannot execute padding_missing_values()"+\
		              "because value of `method` is invalid.")
				return None
		else:
			print("[WARNING] Cannot execute padding_missing_values()"+\
		          "because data-type of `data` is unknown.")
			return None
		
	'''
	  Pre-process labels
	'''
	def get_label_mappings(self, data):
		if type(data).__name__ == "DataFrame":
			class_mapping = {label: idx
					 for idx, label in 
					     enumerate(np.unique(data.iloc[:, -1]))}
			inv_class_mapping = {v: k for k, v in class_mapping.items()}
			label_mappings = class_mapping, inv_class_mapping
			return label_mappings
		else:
			print("[WARNING] Cannot execute get_label_mappings()"+\
		          "because data-type of `data` is unknown.")
			return None
	
	def encode_labels(self, data, label_mappings, method):
		if type(data).__name__ == "DataFrame":
			if method in ("binarize", "categorize"):
				if method == "binarize":
					axis = 0
				else:
					axis = 1
				data.iloc[:, -1] = data.iloc[:, -1].map(label_mappings[axis])
				return data
			else:
				print("[WARNING] Cannot execute encode_labels()"+\
		              "because value of `method` is invalid.")
				return None
		else:
			print("[WARNING] Cannot execute encode_labels()"+\
		          "because data-type of `data` is unknown.")
			return None
	
	'''
	  Pre-process Categorical > "Ordinal" feature
	'''
	def encode_ordinal_feature(self, data, col_name, ordinal_mapping):
		#size_mapping = {"XL":3, 'L':2, 'M':1, 'S':0}
		tmp_data = data[:]
		tmp_data[col_name] = tmp_data[col_name].map(ordinal_mapping)
		return tmp_data
	
	'''
	  Pre-process Categorical > "Ordinal" feature
	'''
	def encode_nominal_feature(self, data, col_name):
		encoded_data = None
		tmp_data = data[:]
		#print(tmp_data)
		'''
		e.g.
		   color size  price class_label
		0  green    M   10.1     T-shirt
		1    red    L   13.5     T-shirt
		2   blue   XL   15.3      jacket
		'''
		color_onehot = OneHotEncoder().fit_transform(tmp_data.loc[:, col_name].values.reshape(-1,1)).toarray()
		del tmp_data[col_name]
		
		encoded_data = [np.hstack((color_onehot[i], row))
				        for i, row in enumerate(tmp_data.to_numpy())]
		'''
		e.g.
		[[0.0 1.0 0.0 'M' 10.1 'T-shirt']
		 [0.0 0.0 1.0 'L' 13.5 'T-shirt']
		 [1.0 0.0 0.0 'XL' 15.3 'jacket']]
		'''
		return np.array(encoded_data)
	
	'''
	  Spliting label & features
	'''
	def label_features_split(self, data, label_col_name):
		X = data.loc[:, data.columns != label_col_name].values
		Y = data.loc[:, label_col_name].values
		return X, Y
	
	'''
	  Feature regularization (normalization / standardization)
	'''
	def regularization(self, X_train, X_test, type_):
		#X_train_copy = X_train[:]
		#X_test_copy = X_test[:]
		if type_ == "standardize":
			for axis in range(X_train.shape[1]):
				col_data = X_train[:, axis]
				mean_ = col_data.mean()
				std_ = col_data.std()
				X_train[:, axis] = (col_data - mean_) / std_
				X_test[:, axis] = (X_test[:, axis] - mean_) / std_
			return X_train, X_test
		
		elif type_ == "normalize":
			for axis in range(X_train.shape[1]):
				col_data = X_train[:, axis]
				max_ = col_data.max()
				min_ = col_data.min()
				X_train[:, axis] = (col_data - min_) / (max_ - min_)
				X_test[:, axis] = (X_test[:, axis] - min_) / (max_ - min_)
			return X_train, X_test
	
	""" Pre-process model """
	def train_logistic_model(self, X_train, X_test, Y_train, Y_test):
		lr = LogisticRegression(penalty="l1", 
						        solver="liblinear",
								C=1.0,
								multi_class="ovr")
		lr.fit(X_train, Y_train)
		get_percentage = lambda num: round(num*100, 4)
		acc_train = get_percentage(lr.score(X_train, Y_train))
		acc_test = get_percentage(lr.score(X_test, Y_test))
		print(f"Training acc: {acc_train}%\n"+\
		      f"Test acc: {acc_test}%\n")
		print("Intercepts of class labels:",
		      *(f"- class label: {class_}\n"+\
		        f"- intercept: {round(intercept,4)}\n"
		        for class_, intercept in zip(lr.classes_, lr.intercept_)), sep='\n')
		print("Weight coefficients of class labels",
			  *(f"- class label: {class_}\n"+\
		        f"- weight coefs:\n {coef}\n"
		        for class_, coef in zip(lr.classes_, lr.coef_)), sep='\n')
		
if __name__ == "__main__":
	preprocess_util = Preprocess_Util()
	
	"""csv_data = \
	
	A, B, C, D
	1.0,2.0,3.0,4.0
	5.0,6.0,, 8.0
	10.0,11.0,12.0,
	.replace('\t', '')
	df = pd.read_csv(StringIO(csv_data))
	print(df, '\n')"""
	
	'''
	""" 1. [Test] Missing values pre-processing """
	# Load test data for [Test 1]
	df = Data_Fetcher().fetch_data("example-values-01")
	print("[TEST] 1")
	print(f"Raw data:\n{df}\n")
	
	print("[TEST] 1a")
	tmp_df = preprocess_util.del_missing_values(df, "col")
	print(f"Delete columns with null values:\n{tmp_df}\n")
	tmp_df = preprocess_util.del_missing_values(df, "row")
	print(f"Delete rows with null values:\n{tmp_df}\n")
	print("[TEST] 1b")
	tmp_df = preprocess_util.padding_missing_values(df, "mean-pandas")
	print(f"Padding missing values [pandas]:\n{tmp_df}\n")
	tmp_df = preprocess_util.padding_missing_values(df, "mean-sklearn")
	print(f"Padding missing values [sklearn]:\n{tmp_df}\n")
	del df
	
	""" 2. [Test] Labels pre-processing """
	# Load test data for [Test 2]
	data_fetcher = Data_Fetcher()
	df = data_fetcher.fetch_data("apparel-data-01")
	print("[TEST] 2")
	print(f"Raw data:\n{df}\n")
	
	label_mappings = preprocess_util.get_label_mappings(df)
	"""
	df[label_col_name] = df[label_col_name].map(class_mapping)
	print(f"After label binarizing:\n {df}\n")
	
	df[label_col_name] = df[label_col_name].map(inv_class_mapping)
	print(f"Before label binarizing:\n {df}\n")"""
	
	print("[TEST] 2a")
	tmp_df = preprocess_util.encode_labels(df, label_mappings, "binarize")
	print(f"After binarizing labels:\n{tmp_df}\n")
	tmp_df = preprocess_util.encode_labels(df, label_mappings, "categorize")
	print(f"Before binarizing labels:\n{tmp_df}\n")

	print("[TEST] 2b")
	ordinal_mapping = {"XL":4, 'L':3, 'M':2, 'S':1}
	col_name = "size"
	tmp_df = preprocess_util.encode_ordinal_feature(df, col_name, ordinal_mapping)
	print(f"After encoding \"size\" column [ordinal]:\n{tmp_df}\n")
	
	print("[TEST] 2c")
	col_name = "color"
	tmp_df = preprocess_util.encode_nominal_feature(df, col_name)
	print(f"After encoding \"color\" column [nominal]:\n{tmp_df}\n")
	
	print("[TEST] 2a/b/c")
	col_name = "size"
	tmp_df = preprocess_util.encode_ordinal_feature(df, col_name, ordinal_mapping)
	tmp_df = preprocess_util.encode_labels(tmp_df, label_mappings, "binarize")
	col_name = "color"
	tmp_df = preprocess_util.encode_nominal_feature(tmp_df, col_name)
	print(f"After pre-processing labela & features:\n{tmp_df}\n")
	print(df, '\n')
	'''
	
	df = Data_Fetcher().fetch_data("wine-data-01")
	""" Get features & labels, and split them into training set & test set """
	label_col_name = "Class label"
	X, Y = preprocess_util.label_features_split(df, label_col_name)
	test_dataset_ratio = 0.3
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	
	""" Regularize features """
	'''
	# method-1
	ss = StandardScaler()
	ss.fit(X_train) # calculate statistics for normalization
	X_train, X_test = ss.transform(X_train), ss.transform(X_test)
	'''
	# method-2
	#regularize_type = "normalize"
	regularize_type = "standardize"
	X_train, X_test = preprocess_util.regularization(X_train, X_test, regularize_type)
	preprocess_util.train_logistic_model(X_train, X_test, Y_train, Y_test)
	