import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline

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

#def plot_data(plot_num, Y_label, X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD):
def plot_data(X, Y, number, save_path, LR, EPOCHS, RANDOM_SEED, THRESHOLD):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
	adaline = Adaline()

	adaline.fit(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
	costs = adaline.costs
	#print(f"len(costs): {len(costs)}")
	#print(costs)
	ax[0].plot(range(1, len(costs)+1), np.log10(costs), marker="o")
	ax[0].set_xlabel("Epochs")
	ax[0].set_ylabel("log(SSE)")
	ax[0].set_title(f"Adaline - Learning rate: {LR}")

	adaline.fit(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
	costs = adaline.costs
	#print(f"len(costs): {len(costs)}")

	ax[1].plot(range(1, len(costs)+1), costs, marker="o")
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("SSE")
	ax[1].set_title(f"Adaline - Learning rate: {LR}")
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.show()

	'''LR = 1e-3
	adaline.fit(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
	costs = adaline.costs
	print(f"len(costs): {len(costs)}")
	ax[1].plot(range(1, len(costs)+1), costs, marker="o")
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("SSE")
	ax[1].set_title(f"Adaline - Learning rate: {LR}")
	plt.tight_layout()
	# plt.savefig("./adaline_1.png", dpi=300)
	plt.show()'''

def get_LR_points(H, L, n_sections, i):
	increment = (H - L) / (n_sections - 1)
	'''
	LR_points = [round(H - increment * i, 1+len(str(H)[str(H).index('.')+1:]))
			     for i in range(n_sections)]
	'''
	LR_points = [round(H - increment * _,
				 i+len(str(int(H/L)))-1-(n_sections-1-_)*int(len(str(int(H/L)))-1!=0))
			     for _ in range(n_sections)]
	print(LR_points)

if __name__ == "__main__":

	EPOCHS = 20
	RANDOM_SEED = 1
	THRESHOLD = 0
	X, Y = load_data()

	'''
	get_LR_points(1e-1, 1e-5, 5, 1)
	get_LR_points(9e-4, 1e-4, 5, 4)
	get_LR_points(7e-4, 5e-4, 5, 4)
	get_LR_points(50e-5, 55e-5, 5, 5)
	get_LR_points(50e-5, 51e-5, 5, 6)
	get_LR_points(500e-6, 503e-6, 5, 7)
	get_LR_points(500e-6, 503e-6, 10, 7)
	'''


	#LRs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	#LRs = [round(i*1e-4,4) for i in range(9, 0, -2)]
	#LRs = [0.0007, 0.00065, 0.0006, 0.00055, 0.0005]
	#LRs = [0.0005, 0.00051, 0.00053, 0.00054, 0.00055]
	#LRs = [0.0005, 0.000503, 0.000505, 0.000508, 0.00051]
	LRs = [0.0005, 0.0005003, 0.0005007, 0.000501, 0.0005013,
		   0.0005017, 0.000502, 0.0005023, 0.0005027, 0.000503]

	for number, LR in enumerate(LRs):
		save_path = f"res/6/Adaline___No.{number+1}___LR={LR}___EPOCHS={EPOCHS}.png"
		plot_data(X, Y, number, save_path, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
