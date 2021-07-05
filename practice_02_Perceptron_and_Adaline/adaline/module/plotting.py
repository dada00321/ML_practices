import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# =============================================================================
# Util-1: Plotting colorful decision regioons
# main func: `plot_classification`
# =============================================================================
def __get_color_tools(Y):
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    return colors, cmap

def __plot_dicision_boundary_with_colors(colors, cmap, W, X, Y, classifier, resolution=0.02):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    X = np.array([xx1.ravel(), xx2.ravel()])
    Z = classifier.net_input(X, W)
    #Z = classifier.activation(net_input)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

def plot_classification(X, Y, classifier,
						classifier_name='',
						weight_update_method='',
						x_label='',
						y_label='',
						save_path=''):
	''' 1. Plot the dicision boundary with colors '''
	colors, cmap = __get_color_tools(Y)
	__plot_dicision_boundary_with_colors(colors, cmap, classifier.W[1:], X, Y, classifier)

	''' 2. Plot data points '''
	plt.scatter(X[:50, 0], X[:50, 1],
	            color="red", marker="o", label="setosa")
	plt.scatter(X[50:100, 0], X[50:100, 1],
	            color="blue", marker="x", label="versicolor")

	''' 3. Set the classification map '''
	empty_choices = ("", None)
	# Set title
	if classifier_name in empty_choices:
		classifier_name = "(classifier name)"
	if weight_update_method in empty_choices:
		weight_update_method = "(weight update method)"
	plt.title(f"{classifier_name} - {weight_update_method}")

	# Set x & y labels
	if x_label in empty_choices:
		x_label = "(classification - x label)"
	if y_label in empty_choices:
		y_label = "(classification - y label)"
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	plt.legend(loc="upper left")
	plt.tight_layout()

	# Save the map to `save_path` if given
	if save_path not in empty_choices:
		if os.path.exists(save_path):
			plt.savefig(save_path, dpi=300)
		else:
			print("[WARNING] The path `save_path` does not exist.")

	''' 4. Show the classification map '''
	plt.show()

# =============================================================================
# Util-2: Plotting loss/cost
# main func: `plot_loss`
# =============================================================================
def plot_loss(classifier):
	losses = classifier.costs
	plt.plot(range(1, len(losses)+1), losses, marker='o')
	plt.title("Adaline - Gradient Descent")
	plt.xlabel("Epochs")
	plt.ylabel("SSE")
	plt.tight_layout()
	# plt.savefig('./XX.png', dpi=300)
	plt.show()
