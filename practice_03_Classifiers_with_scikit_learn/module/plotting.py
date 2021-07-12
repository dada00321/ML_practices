import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

class Plotting:
	def __init__(self, classifier, classifier_name,
				 x_label, y_label,
				 scatter_name_dict=None,
				 lr=None,
				 multi_class=None):
		empty_choices = ('', None)
		self.set_classifier(classifier)

		#if classifier_name in empty_choices:
		#	classifier_name = "(classifier name)"
		self.classifier_name = classifier_name

		if x_label in empty_choices:
			x_label = "(classification - x label)"
		self.x_label = x_label

		if y_label in empty_choices:
			y_label = "(classification - y label)"
		self.y_label = y_label

		self.scatter_name_dict = scatter_name_dict

		title = f"{classifier_name}\n"
		if lr is not None:
			self.LR = lr
			title += f"Learning rate: {lr}"
		if multi_class is not None:
			self.MULTI_CLASS = multi_class
			title += f"multi class: {multi_class}"
		self.title = title

	def set_classifier(self, classifier):
		self.classifier = classifier

	# =============================================================================
	# Util-1: Plotting colorful decision regioons
	# main func: `plot_classification`
	# =============================================================================
	def __get_color_tools(self, Y):
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ("red", "blue", "lightgreen", "gray", "cyan")
		cmap = ListedColormap(colors[:len(np.unique(Y))])
		return markers, colors, cmap

	def __plot_dicision_boundary_with_colors(self, colors, cmap, X, Y, resolution=0.02):
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
							   np.arange(x2_min, x2_max, resolution))
		#X = np.array([xx1.ravel(), xx2.ravel()])
		#Z = self.classifier.net_input(X, W)
		#Z = self.classifier.activation(net_input)
		Z =  self.classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())

	def __save_fig_to_path(self, plt, save_path):
		# Save the figure to `save_path` if given
		if save_path not in (None, ''):
			save_dir = '/'.join(save_path.split('/')[:-1])
			if os.path.exists(save_dir):
				plt.savefig(save_path, dpi=300)
			else:
				print("[WARNING] The parent direnctory of `save_path` does not exist.")
		else:
			print("[WARNING] `__save_fig_to_path()`: The parameter `save_path` is invalid.")

	def __save_polt_and_show(self, plt, save_path): # save_path: if need to save figure
		plt.tight_layout()
		self.__save_fig_to_path(plt, save_path) # Save the figure to `save_path` if given
		plt.show()

	def plot_classification(self,
							X, Y,
							save_path='',
							test_idx=None,
							test_label=[None]):
		''' Note: The classifier `classifier` had to be trained! '''
		''' 1. Plot the dicision boundary with colors '''
		markers, colors, cmap = self.__get_color_tools(Y)
		self.__plot_dicision_boundary_with_colors(colors, cmap, X, Y)

		''' 2. Plot data points '''
		#print(X.shape); print(Y.shape)
		labels = list(self.scatter_name_dict.values())
		for idx, cl in enumerate(np.unique(Y)):
			plt.scatter(X[Y==cl, 0],
				        X[Y==cl, 1],
			            alpha=0.8,
						color=cmap(idx),
						edgecolor="black",
					    marker=markers[idx],
						label=labels[idx])

		# Highlight test samples if input parameter `test_idx` is not None
		if test_idx is not None:
			#X_test, Y_test = X[test_idx, :], Y[test_idx]
			X_test = X[test_idx, :]
			plt.scatter(X_test[:, 0],
					    X_test[:, 1],
					    c='',
						edgecolor="black",
					    alpha=1.0,
						linewidth=1,
						marker='o',
						s=100,
						label="test dataset")

		''' 3. Set the classification map '''
		plt.title(self.title)
		plt.xlabel(self.x_label)
		plt.ylabel(self.y_label)
		plt.legend(loc="upper left")

		''' 4. Show the classification map '''
		self.__save_polt_and_show(plt, save_path)