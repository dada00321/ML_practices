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
def plot_data(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD):
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
	# plt.savefig(f"./adaline_1_LR={LR}.png", dpi=300)

	adaline.fit(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
	costs = adaline.costs
	#print(f"len(costs): {len(costs)}")
	ax[1].plot(range(1, len(costs)+1), costs, marker="o")
	ax[1].set_xlabel("Epochs")
	ax[1].set_ylabel("SSE")
	ax[1].set_title(f"Adaline - Learning rate: {LR}")
	plt.tight_layout()
	# plt.savefig(f"./adaline_2_LR={LR}.png", dpi=300)
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

if __name__ == "__main__":
	EPOCHS = 20
	RANDOM_SEED = 1
	THRESHOLD = 0
	X, Y = load_data()

	LRs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	for LR in LRs:
		plot_data(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
