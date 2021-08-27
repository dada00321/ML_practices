from sklearn.svm import SVC

# =============================================================================
# for loading data
# =============================================================================
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from module.plotting import Plotting

# =============================================================================
# for plotting & observing regularization parameter
# =============================================================================
import matplotlib.pyplot as plt

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

def generate_nonlinear_example_data(is_show=True):
	X_xor = np.random.randn(200, 2)
	Y_xor = np.logical_xor(X_xor[:, 0] > 0,
						   X_xor[:, 1] > 0)
	Y_xor = np.where(Y_xor, 1, -1)
	if is_show:
		plt.scatter(X_xor[Y_xor==1, 0],
				    X_xor[Y_xor==1, 1],
					c='b', marker='x',
					label='1')
		plt.scatter(X_xor[Y_xor==-1, 0],
				    X_xor[Y_xor==-1, 1],
					c='r', marker='s',
					label='-1')
		plt.xlim([-3,3])
		plt.ylim([-3,3])
		plt.legend(loc="best")

		plt.tight_layout()
		plt.show()
	return X_xor, Y_xor

def train_linear_SVM(X_train, X_test, Y_train, Y_test, sample_size):
	KERNEL = "linear"
	RAMDOM_SEED = 1
	#C = 100.0
	C = 10.0
	#C = 1.0
	svm = SVC(kernel=KERNEL, C=C, random_state=RAMDOM_SEED)
	svm.fit(X_train, Y_train)

	# plot classification result
	classifier = svm
	classifier_name = f"SVM (C={C})"
	x_label = "petal length (cm)"
	y_label = "petal width (cm)"
	scatter_name_dict = {0: "Setosa",
					     1: "Versicolor",
						 2: "Virginica"}
	plotting = Plotting(classifier,
					    classifier_name,
					    x_label,
						y_label,
						scatter_name_dict)
	X_combined = np.vstack((X_train, X_test))
	Y_combined = np.hstack((Y_train, Y_test))
	save_path = "res/sklearn_SVM/"+\
				f"linear_SVM/sklearn_linear_SVM___C={C}.png"
	test_idx = range(int(sample_size*(1-test_dataset_ratio)), sample_size)
	plotting.plot_classification(X_combined, Y_combined,
							     save_path, test_idx)

def train_kernel_SVM(X_train, X_test, Y_train, Y_test, sample_size, test_dataset_ratio, GAMMA_=0.1):
	KERNEL = "rbf"
	RAMDOM_SEED = 1
	#GAMMA = 0.1
	GAMMA = GAMMA_

	#C = 100.0
	C = 10.0
	#C = 1.0
	svm = SVC(kernel=KERNEL, C=C, random_state=RAMDOM_SEED, gamma=GAMMA)
	svm.fit(X_train, Y_train)

	# plot classification result
	classifier = svm
	classifier_name = f"SVM (C={C}, Gamma={GAMMA})"
	x_label = "petal length (cm)"
	y_label = "petal width (cm)"
	scatter_name_dict = {0: "Setosa",
					     1: "Versicolor",
						 2: "Virginica"}
	plotting = Plotting(classifier,
					    classifier_name,
					    x_label,
						y_label,
						scatter_name_dict)
	X_combined = np.vstack((X_train, X_test))
	Y_combined = np.hstack((Y_train, Y_test))
	save_path = "res/sklearn_SVM/"+\
				f"kernel_SVM/sklearn_linear_SVM___C={C}_Gamma={GAMMA}.png"
	test_idx = range(int(sample_size*(1-test_dataset_ratio)), sample_size)
	plotting.plot_classification(X_combined, Y_combined,
							     save_path, test_idx)

if __name__ == "__main__":
	X, Y = load_data("iris")
	sample_size = np.bincount(Y).sum()
	test_dataset_ratio = 0.3
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)

	# 3.10
	#train_linear_SVM(X_train, X_test, Y_train, Y_test, sample_size)

	# 3.11
	#plot_nonlinear_example_data()

	# 3.12 -1
	'''
	#X_xor, Y_xor = generate_nonlinear_example_data(False)
	X_xor, Y_xor = generate_nonlinear_example_data()
	train_kernel_SVM(X_xor, X_test, Y_xor, Y_test, sample_size)
	'''

	# 3.12 -2
	for GAMMA_ in [0.1, 0.2]:
	#for GAMMA_ in [0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
		train_kernel_SVM(X_train, X_test, Y_train, Y_test, sample_size, test_dataset_ratio, GAMMA_)
