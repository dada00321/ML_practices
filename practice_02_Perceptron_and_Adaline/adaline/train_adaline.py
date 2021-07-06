import pandas as pd
import numpy as np
from adaline_GD import Adaline_GD
from adaline_SGD import Adaline_SGD
from module.plotting import Plotting
import os
#import time

def load_data():
    iris_data_download_link = "".join(["https://archive.ics.uci.edu/ml/",
                                       "machine-learning-databases",
                                       "/iris/iris.data"])
    #print(iris_data_download_link)
    df = pd.read_csv(iris_data_download_link, header=None, encoding="utf-8")
    #print(df.tail())

    """
    Select setosa(山鳶尾) and versicolor(變色鳶尾)
     - as 2 classes to practice classification
     - [sample size = 100]
    """
    Y = df.iloc[0:100, 4].values
    Y = np.where(Y == "Iris-setosa", -1, 1)

    # Extract sepal(花萼長) length and petal(花瓣長) length
    # => as 2 input features/neurons
    X = df.iloc[0:100, [0, 2]].values
    return X, Y

def standardize_features(X):
	X_standardized = np.copy(X)
	standardize = lambda x: (x- np.mean(x)) / np.std(x)
	X_standardized[:,0] = standardize(X_standardized[:,0])
	X_standardized[:,1] = standardize(X_standardized[:,1])
	return X_standardized

def train(X, Y, mode, LR, EPOCHS, RANDOM_SEED, THRESHOLD):
	supported_mode = ("GD", "SGD")
	if mode in supported_mode:
		if mode == "GD":
			adaline = Adaline_GD()
		elif mode == "SGD":
			adaline = Adaline_SGD()
		X_standardized = standardize_features(X)
		adaline.fit(X_standardized, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
		return adaline
	else:
		print("[WARNING] The parameter `mode` is invalid in train()")
		return None

def __get_LR_points(H, L, n_sections, i):
	increment = (H - L) / (n_sections - 1)
	'''
	LR_points = [round(H - increment * i, 1+len(str(H)[str(H).index('.')+1:]))
			     for i in range(n_sections)]
	'''
	LR_points = [round(H - increment * _,
				 i+len(str(int(H/L)))-1-(n_sections-1-_)*int(len(str(int(H/L)))-1!=0))
			     for _ in range(n_sections)]
	print(LR_points)

def __create_dir_if_not_exists(path):
	if not os.path.exists(path):
		os.mkdir(path)

if __name__ == "__main__":
	# =============================================================================
	# 1. Setting the training hyperparameters
	# =============================================================================
	EPOCHS = 20
	RANDOM_SEED = 1
	THRESHOLD = 0
	X, Y = load_data()

	#mode = "GD"
	mode = "SGD"

	'''
	__get_LR_points(1e-1, 1e-5, 5, 1)
	__get_LR_points(9e-4, 1e-4, 5, 4)
	__get_LR_points(7e-4, 5e-4, 5, 4)
	__get_LR_points(50e-5, 55e-5, 5, 5)
	__get_LR_points(50e-5, 51e-5, 5, 6)
	__get_LR_points(500e-6, 503e-6, 5, 7)
	'''

	# =============================================================================
	# 2. Training model and plotting the classification result
	# =============================================================================
	classifier_name = "Adaline"
	if mode == "GD":
		weight_update_method = "Gradient Descent"
	elif mode == "SGD":
		weight_update_method = "Stochastic Gradient Descent"
	x_label = "sepallength (cm)"
	y_label = "petallength (cm)"
	scatter_name_1 = "setosa"
	scatter_name_2 = "versicolor"

	#----------------
	# 2-1 Train & plot losses / costs of classification models
	#----------------
	LRs_dict = dict()
	LRs_dict['1'] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	LRs_dict['2'] = [round(i*1e-4,4) for i in range(9, 0, -2)]
	LRs_dict['3'] = [0.0007, 0.00065, 0.0006, 0.00055, 0.0005]
	LRs_dict['4'] = [0.0005, 0.00051, 0.00053, 0.00054, 0.00055]
	LRs_dict['5'] = [0.0005, 0.000503, 0.000505, 0.000508, 0.00051]
	LRs_dict['6'] = [0.0005, 0.0005003, 0.0005007, 0.000501, 0.0005013, 0.0005017, 0.000502, 0.0005023, 0.0005027, 0.000503]

	plot_types = {1: "SSE", 2: "SSE+log SSE"}
	testing_plot_type = plot_types[2]

	for test_group_number, LRs in LRs_dict.items():
		print(f"Start to training & plotting group {test_group_number}...")
		for sample_number, LR in enumerate(LRs):
			print(f" {sample_number+1:2d}. The sample with LR = {LR} is runnuing!")
			adaline = train(X, Y, mode, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
			if adaline is not None:
				plotting = Plotting(adaline, classifier_name, weight_update_method,
								    x_label, y_label, scatter_name_1, scatter_name_2)
				loss_plot_dir = f"./res/adaline_loss/{test_group_number}"
				loss_plot_img_dir = loss_plot_dir + f"/with standardize___{testing_plot_type}"
				loss_plot_path = loss_plot_img_dir + f"/AdalineGD___No.{sample_number+1}___mode={mode}___LR={LR}___EPOCHS={EPOCHS}.png"
				__create_dir_if_not_exists(loss_plot_dir)
				__create_dir_if_not_exists(loss_plot_img_dir)
				plotting.plot_costs(testing_plot_type, loss_plot_path)
				print()
		#time.sleep(5)

	#----------------
	# 2-2 Train & plot decision regoins and data points
	#----------------
	'''
	LR = 1e-2
	adaline = train(X, Y, mode, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
	if adaline is not None:
		plotting = Plotting(adaline, classifier_name, weight_update_method,
						    x_label, y_label, scatter_name_1, scatter_name_2)
		save_path = "res/adaline_classification/"+\
			        f"adaline_classification___mode={mode}___LR={LR}___EPOCHS={EPOCHS}.png"
		plotting.plot_classification(X, Y, save_path=save_path)
	'''