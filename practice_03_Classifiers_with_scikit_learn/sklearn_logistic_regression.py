# =============================================================================
# for loading data
# =============================================================================
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from module.plotting import Plotting
# =============================================================================
# for training data
# =============================================================================
from sklearn.linear_model import LogisticRegression

def show_sample_size_of_each_class(Y, title):
	'''
	Y: 1-dimensional data | e.g. shape: (100,)
	'''
	if len(Y) > 0:
		classes = np.unique(Y)
		sample_sizes = np.bincount(Y)
		tmp = dict(zip(classes, sample_sizes))
		print(f"\n------ {title} ------")
		print(*(f"class: {class_}\t"+\
			    f"sample size: {tmp[class_]}"
				for class_ in tmp.keys()), sep='\n')
	else:
		print("[WARNING] The parameter `Y` of function:"+\
		      " `show_sample_size_of_each_class()` is empty.")
		return None

def load_data(type_):
	if type_ == "iris":
		iris_data = datasets.load_iris()
		X = iris_data.data[:, [2, 3]]
		Y = iris_data.target
		return X, Y
	else:
		print("[WARNING] The parameter `type_` of function:"+\
		      " `load_data()` is undefined.")
		return None

def normalize_input_features(X_train, X_test):
	'''
	rtn_val: normalized X_train and X_test (via same statistics)
	'''
	ss = StandardScaler()
	ss.fit(X_train) # calculate statistics for normalization
	return ss.transform(X_train), ss.transform(X_test)

if __name__ == "__main__":
	X, Y = load_data("iris")
	show_sample_size_of_each_class(Y, "original data")

	sample_size = np.bincount(Y).sum()
	test_dataset_ratio = 0.3
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	show_sample_size_of_each_class(Y_train, "training data")
	show_sample_size_of_each_class(Y_test, "test data")

	X_train, X_test = normalize_input_features(X_train, X_test)

	#LR = 0.1
	RAMDOM_SEED = 1
	C = 100.0
	SOLVER = "lbfgs"
	MULTI_CLASS  = "ovr"
	#MULTI_CLASS  = "multinomial"
	# 1. construct model
	logistic_regression = LogisticRegression(C = C, random_state=RAMDOM_SEED,
										     solver=SOLVER,
											 multi_class=MULTI_CLASS)
	# 2. train
	logistic_regression.fit(X_train, Y_train)

	# 3. classify
	y_pred = logistic_regression.predict(X_test)
	num_of_correct, num_of_incorrect = np.bincount((y_pred != Y_test))
	#print(num_of_correct, num_of_incorrect)
	accuracy = round(num_of_correct*100 / (num_of_correct + num_of_incorrect), 3)
	print(f"\naccuracy: {accuracy}%")
	accuracy = round(logistic_regression.score(X_test, Y_test), 3)
	print(f"\naccuracy: {accuracy}")

	# 4. plot classification result
	classifier = logistic_regression
	classifier_name = "Logistic Regression"
	x_label = "petal length (cm)"
	y_label = "petal width (cm)"
	scatter_name_dict = {0: "Setosa",
					     1: "Versicolor",
						 2: "Virginica"}
	plotting = Plotting(classifier,
					    classifier_name,
					    x_label,
						y_label,
						scatter_name_dict,
						multi_class=MULTI_CLASS)
	X_combined = np.vstack((X_train, X_test))
	Y_combined = np.hstack((Y_train, Y_test))
	save_path = "res/sklearn_perceptron_classification/"+\
				f"sklearn_perceptron_classification___C={C}_"+\
				f"__multi_class={MULTI_CLASS}.png"
	test_idx = range(int(sample_size*(1-test_dataset_ratio)), sample_size)
	plotting.plot_classification(X_combined, Y_combined,
							     save_path, test_idx)

	print(X_test[:3, :])
	print('\n', logistic_regression.predict_proba(X_test[:3, :]), sep='')
	print('\n', logistic_regression.predict_proba(X_test[:3, :]).argmax(axis=1), sep='')
	print('\n', logistic_regression.predict(X_test[:3, :]), sep='')
	print('\n', X_test[0, :], sep='')
	print('\n', X_test[0, :].reshape(1, -1), sep='')
	#print('\n', logistic_regression.predict(X_test[:3, :]), sep='')