import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

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

def plot_data(X, Y):
    plt.scatter(X[:50, 0], X[:50, 1],
                color="red", marker="o", label="setosa")
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color="blue", marker="x", label="versicolor")

    plt.xlabel("petal length [cm]")
    plt.ylabel("sepal length [cm]")
    plt.legend(loc="upper left")

    plt.tight_layout()
    # plt.savefig("./iris_1.png", dpi=300)
    plt.show()

def train(perceptron, X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD):

    print(X.shape)

    X = perceptron.fit(X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)

    plt.plot(range(1, len(perceptron.errors_in_epochs) + 1),
             perceptron.errors_in_epochs, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of misclassifications")

    plt.tight_layout()
    # plt.savefig("./perceptron_1.png", dpi=300)
    plt.show()
    return X

def plot_decision_regions(W, X, Y, classifier, resolution=0.02):
    # setup marker generator and color map
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(Y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    X = np.array([xx1.ravel(), xx2.ravel()])
    net_input = classifier.net_input(W, X)
    Z = classifier.activation(net_input)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plot_data()

if __name__ == "__main__":
    X, Y = load_data()
    plot_data(X, Y)
    perceptron = Perceptron()
    LR = 1e-2
    EPOCHS = 10
    RANDOM_SEED = 1
    THRESHOLD = 0
    train(perceptron, X, Y, LR, EPOCHS, RANDOM_SEED, THRESHOLD)
    plot_decision_regions(perceptron.W[1:], X, Y, classifier=perceptron)
