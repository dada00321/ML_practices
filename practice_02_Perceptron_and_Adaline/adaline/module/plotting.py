import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

class Plotting:
	def __init__(self, classifier, classifier_name, weight_update_method,
			     x_label, y_label, scatter_name_1, scatter_name_2):
		empty_choices = ('', None)
		self.set_classifier(classifier)

		#if classifier_name in empty_choices:
		#	classifier_name = "(classifier name)"
		self.classifier_name = classifier_name

		if weight_update_method in empty_choices:
			weight_update_method = "(weight update method)"
		self.weight_update_method = weight_update_method

		if x_label in empty_choices:
			x_label = "(classification - x label)"
		self.x_label = x_label

		if y_label in empty_choices:
			y_label = "(classification - y label)"
		self.y_label = y_label

		title = f"{classifier_name} - {weight_update_method}\n"+\
			    f"Learning rate: {classifier.LR}   "+\
				f"Epochs: {classifier.EPOCHS}"
		self.title = title

		if scatter_name_1 in empty_choices:
			scatter_name_1 = "(scatter name 1)"
		self.scatter_name_1 = scatter_name_1

		if scatter_name_2 in empty_choices:
			scatter_name_2 = "(scatter name 2)"
		self.scatter_name_2 = scatter_name_2

	def set_classifier(self, classifier):
		self.classifier = classifier

	# =============================================================================
	# Util-1: Plotting colorful decision regioons
	# main func: `plot_classification`
	# =============================================================================
	def __get_color_tools(self, Y):
	    colors = ("red", "blue", "lightgreen", "gray", "cyan")
	    cmap = ListedColormap(colors[:len(np.unique(Y))])
	    return colors, cmap

	def __plot_dicision_boundary_with_colors(self, colors, cmap, W, X, Y, resolution=0.02):
	    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	                         np.arange(x2_min, x2_max, resolution))
	    X = np.array([xx1.ravel(), xx2.ravel()])
	    Z = self.classifier.net_input(X, W)
	    #Z = classifier.activation(net_input)
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
							SAMPLE_SIZE,
							save_path=''):
		''' Note: The classifier `classifier` had to be trained! '''

		''' 1. Plot the dicision boundary with colors '''
		colors, cmap = self.__get_color_tools(Y)
		W = self.classifier.W[1:]
		self.__plot_dicision_boundary_with_colors(colors, cmap, W, X, Y)

		''' 2. Plot data points '''
		break_point = int(SAMPLE_SIZE/2)
		plt.scatter(X[:break_point, 0], X[:break_point, 1],
		            color="red", marker="o", label=self.scatter_name_1)
		plt.scatter(X[break_point:SAMPLE_SIZE, 0], X[break_point:SAMPLE_SIZE, 1],
		            color="blue", marker="x", label=self.scatter_name_2)

		''' 3. Set the classification map '''
		plt.title(self.title)
		plt.xlabel(self.x_label)
		plt.ylabel(self.y_label)
		plt.legend(loc="upper left")

		''' 4. Show the classification map '''
		self.__save_polt_and_show(plt, save_path)

	# =============================================================================
	# Util-2: Plotting loss/cost
	# main func: `plot_costs`
	# =============================================================================
	def plot_costs(self,
				   plot_type='',
				   save_path=''):
		if plot_type not in ("SSE", "SSE+log SSE"):
			print("[WARNING] The parameter `plot_type` of function `plot_costs` is invalid.")
		else:
			costs = self.classifier.costs

			if plot_type == "SSE":
				plt.plot(range(1, self.classifier.EPOCHS+1), costs, marker='o')
				plt.title(self.title)
				plt.xlabel("Epochs")
				plt.ylabel("SSE")
				#plt.ylabel("Avgerage Cost")

			else: # (plot_type == "SSE + log SSE")
				fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
				#for i in range(len(costs)):
					#costs[i][costs[i]<1] = 1
				log_costs = np.log10(costs)
				ax[0].plot(range(1, self.classifier.EPOCHS+1), log_costs, marker='o')
				ax[0].set_title(self.title)
				ax[0].set_xlabel("Epochs")
				ax[0].set_ylabel("log(SSE)")
				#ax[0].set_ylabel("log(Avgerage Cost)")

				ax[1].plot(range(1, self.classifier.EPOCHS+1), costs, marker='o')
				ax[1].set_title(self.title)
				ax[1].set_xlabel("Epochs")
				ax[1].set_ylabel("SSE")
				#ax[1].set_ylabel("Avgerage Cost")
			self.__save_polt_and_show(plt, save_path)
