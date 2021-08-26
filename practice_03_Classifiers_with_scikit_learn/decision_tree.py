import numpy as np

# =============================================================================
# for Scikit-learn Decision Tree
# =============================================================================
from sklearn import datasets
from sklearn import tree as sklearn_tree_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
# for Plotting
# =============================================================================
from module.plotting import Plotting
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

class Decision_Tree_Generator():
	def load_data(self, dataset):
		if dataset in ("iris", "activity"):
			if dataset == "iris":
				iris_data = datasets.load_iris()
				X = iris_data.data[:, [2, 3]]
				Y = iris_data.target
				return X, Y
			else:
				data = [[2,2,1,0,"yes"],
				        [2,2,1,1,"no"],
						[1,2,1,0,"yes"],
						[0,0,0,0,"yes"],
						[0,0,0,1,"no"],
						[1,0,0,1,"yes"],
						[2,1,1,0,"no"],
						[2,0,0,0,"yes"],
						[0,1,0,0,"yes"],
						[2,1,0,1,"yes"],
						[1,2,0,0,"no"],
						[0,1,1,1,"no"]]
				'''
				data = [[2,2,1,0,"yes"],
				        [2,2,1,1,"yes"],
						[1,2,1,0,"yes"],
						[0,0,0,0,"yes"]]
				
				data = [[2,2,1,0,"no"],
				        [2,2,1,1,"no"],
						[1,2,1,0,"no"],
						[0,0,0,0,"no"]]
				#data = [["no"],["yes"],["yes"],["no"],["yes"]]
				#data = [["no"],["yes"],["yes"],["no"],["no"]]
				'''
				features = ["天氣","溫度","濕度","風速"]
				return data, features
		else:
			print("[WARNING] Unknown dataset.")
			return None
	
	"""
	Handmade decision tree
	---
	main functions: 
		(1) create_tree
		    input: `data` <list> (list of <list>):
				     => "Labels" placed on last column, 
			            "realized feature values" placed on other columns
			       `features` <list> (list of <str>): 
					 => Feature names of "realized feature values"
			
			process: Recursively call itself to split current data by the feature
			         with the greatest Info. Gain in each recursion
					 untill (1) Labels of current data are same, or 
					        (2) All features to split data were be used
			
			output: `decision tree` <dict>
			          => key: can be feature name or realized feature value
					     value: can be temporary dict (<dict>) or class-label (<str>)
					  => example:
						  {'濕度': {0: {'溫度': {0: 'yes', 1: 'no'}}, 1: 'no'}}
		
		(2) list_combinations
		    input: `decision tree` <dict>
			       `delimiter` <str>: 
			         => [optional] default is " => "
				   `prefix` <str>, `combinations` <list>: 
				     => [don't care] used by recursion (internally)
			
			process: Recursively form up list of "decision paths".
			
			output: combinations <list> (list of "decision paths")
				     => example:
						  ['[濕度=0] => [溫度=0] => [yes]', 
		                   '[濕度=0] => [溫度=1] => [no]',
		                   '[濕度=1] => [no]']
		(3) show_decision_tree
		    input: `combinations` <list>
			process: Print out `combinations` sequentially.
			output: X
		
		(4) predict
		    input:  `options` <list> (list of <str>)
					 => "realized feature values"
					`features` <list> (list of <str>)
					 => Feature names of "realized feature values"
					`combinations` <list> (list of "decision paths")
			process: Doing simple string matching to find out the suitable combination
			         for given "realized feature values".
			output: class-label (depends on the result of decision tree)
	"""
	def __get_sorted_occurences(self, labels):
		occurences = dict()
		for label in labels:
			occurences.setdefault(label, 0)
			occurences[label] += 1
		# Sort dict by val of occurence (in desc.)
		sorted_occurences = sorted(occurences.items(),
			                       key=lambda x:x[1],
					               reverse=True)
		# e.g. [('yes', 7), ('no', 5)]
		return sorted_occurences
	
	def __get_majority_label(self, sorted_occurences):
		# Return the 1st element in tuple with the most occurence
		# sorted_occurences[0]: tuple with the most occurence
		# sorted_occurences[0][0]: 1st element (label) in tuple
		# e.g. yes
		return sorted_occurences[0][0]
	
	def __get_probs(self, sorted_occurences):
		# Return elements placed on 2nd order (occurence) in tuples
		occurences = [e[1] for e in sorted_occurences]
		total = sum(occurences)
		probs = [e / total for e in occurences]
		return probs
	
	def __group_by(self, data, feature_i, feature_val_j, del_feature=False):
		rtn_data = list()
		for record in data:
			if record[feature_i] == feature_val_j:
				if del_feature is False:
					# append the whole record
					rtn_data.append(record)
				else:
					# append the whole record except the element placed on feature_i
					rtn_data.append([record[k] for k in range(len(record)) if k != feature_i])
		return rtn_data
		
	def create_tree(self, data, features):
		labels = [record[-1] for record in data]
		
		# case (1): All labels are same
		if labels.count(labels[0]) == len(labels):
			return labels[0]
		
		sorted_occurences = self.__get_sorted_occurences(labels)
		# case (2): All features were be used
		if len(data[0]) == 1:
			# Return the majority label
			return self.__get_majority_label(sorted_occurences)
		
		# case (3): Otherwise
		
		''' Step 1. Define formula of impurity '''
		entropy = lambda probs: -sum([prob * np.log2(prob) for prob in probs])
		
		''' Step 2. Obtain the "best feature" ("feature to split") within current data '''
		best_feature_idx = -1   # initial index of best feature
		greatest_info_gain = 0.0   # initial greatest info. gain
		
		''' Step 2a. Calculate parent impurity '''
		parent_impurity = entropy(self.__get_probs(sorted_occurences))
		# [TEST]
		#print(f"[TEST] parent_impurity: {parent_impurity}\n")
		
		for i, feature in enumerate(features):
			col_i_data = [record[i] for record in data]
			# *** [TEST] ***
			#print(f"[TEST] possible values among feature {i} ({feature}):")
			#print(set(col_i_data), '\n')
			
			''' Step 2b. Calculate total impurity of child nodes '''
			total_child_impurity = 0.0
			N_total = len(col_i_data)
			for j in set(col_i_data):
				N_child = col_i_data.count(j)
				# [TEST]
				#print(f"[TEST] N_child_node_j (j={j}) / N_total_num_of_feature_{i}:")
				#print(N_child / N_total, '\n')
				
				# Group batchs of data by "j" (val. of feature i: j)
				grouped_data = self.__group_by(data, i, j)
				#print(grouped_data)
				labels = [record[-1] for record in grouped_data]
				#print(N_child, N_total)
				#print(self.__get_sorted_occurences(labels), '\n')
				#print(self.__get_probs(self.__get_sorted_occurences(labels)), '\n')
				I_child = entropy(self.__get_probs(self.__get_sorted_occurences(labels)))
				
				# [TEST]
				#print(f"[TEST] (N_child/ N_total) * I_child:")
				#print(N_child / N_total * I_child, '\n')
				total_child_impurity += N_child / N_total * I_child
			# [TEST]
			#print(f"total_child_impurity: {total_child_impurity}\n")
			
			''' Step 2c. Obtain information gain of feature i '''
			# *** [TEST] ***
			info_gain = parent_impurity - total_child_impurity
			#print(f"Info. Gain of feature {i}: {info_gain}\n")
			
			''' Step 2d. Replace the current "best feature index" with index of feature i
			             if info. gain of feature i is the largest one '''
			# *** [TEST] ***
			if info_gain > greatest_info_gain:
				best_feature_idx = i
				greatest_info_gain = info_gain
		
		''' Step 2d. Obtain the best feature '''
		# *** [TEST] ***
		best_feature = features[best_feature_idx]
		#print("[TEST] Feature to split:")
		#print(f"{best_feature} (idx: {best_feature_idx})\n")
		
		''' Step 3. Form up the decision tree '''
		### Need to delete "used feature name" & "col. in data"
		features_copy = features[:]
		del features_copy[best_feature_idx]
		possible_values = set([record[best_feature_idx] for record in data])
		decision_tree = {best_feature: dict()}
		for val in possible_values:
			grouped_data = self.__group_by(data, best_feature_idx, val, True)
			#print(grouped_data)
			#decision_tree[best_feature][val] = dict()
			decision_tree[best_feature][val] = self.create_tree(grouped_data, features_copy)
		#print(decision_tree)
		#print(features)
		return decision_tree
	
	def list_combinations(self, tree, delimiter=" => ", prefix=None, combinations=[]):
		for k, v in tree.items():
			if type(k).__name__ == "str":
				tree = v
				# k is feature (imply: next k (k') is option)
				for option in v:
					#print(f"[{k}={option}]{delimiter}", end='')
					tmp = tree[option]
					msg = f"[{k}={option}]{delimiter}"
					if prefix is not None:
						msg = prefix + msg
					if type(tmp).__name__ == "str":
						#print(f"{msg}[{tmp}]\n", end='') # show a combination
						combinations.append(f"{msg}[{tmp}]")
					else: # dict
						self.list_combinations(tmp, delimiter, msg, combinations)
		return combinations
	
	def show_decision_tree(self, combinations):
		print("[INFO] Possible combinations (rules) of decision tree:")
		print(*(f"{i+1:2d}. {e}" for i, e in enumerate(combinations)), sep='\n', end='\n'*2)
		
	def predict(self, features, options, combinations, delimiter=" => "):
		print(f"[INFO] Predict for feature set: {options} ...\n")
		rtn_list = list()
		matched_idx = -1
		target_set = [f"[{feature}={option}]" for feature, option in zip(features, options)]
		#print(target_set, '\n')
		
		for i, combination in enumerate(combinations):
			feature_vals = combination.split(delimiter)
			if all(e in target_set for e in feature_vals[:-1]):
				#print(combination)
				rtn_list.append(combination)
				matched_idx = i
				break
		
		if len(rtn_list) > 0:
			data_to_predict = " and ".join(target_set)
			print("[INFO] Data to predict:\n", data_to_predict, sep='', end='\n'*2)
			print(f"[INFO] Satisfied rule:\n{matched_idx+1:2d}. {rtn_list[0]}", sep='', end='\n'*2)
			result = rtn_list[0].split(delimiter)[-1][1:-1]
			print(f"[INFO] Result({options}):", result, sep=' ')
			return result
		else:
			print("[INFO] Check whether the data is incomplete.")
			return None
	"""
	Scikit-learn decision tree
	"""
	def __get_classes(self, dataset, type_):
		if dataset == "iris":
			if type_ in ("dict", "list"):
				if type_ == "dict":
					return  {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
				elif type_ == "list":
					return ["Setosa", "Versicolor", "Virginica"]
			else:
				print("[WARNING] Cannot execute func: `get_classes`()"+\
			          " because input parameter: `type_` is invalid.")
				return None
		else:
			print("[WARNING] Unknown dataset.")
			return None
		
	def __get_label_names(self, dataset, remark="with_unit"):
		if dataset == "iris":
			if remark == "with_unit":
				return ["petal length (cm)", "petal width (cm)"]
			elif remark == "without_unit":
				return ["petal length", "petal width"]
			else:
				print("[WARNING] Cannot execute func: `__get_label_names()`"+\
		              " because the input parameter of `remark` is invalid.")
				return None
		else:
			print("[WARNING] Unknown dataset.")
			return None
		
	def create_tree_scikit(self, X_train, X_test, Y_train, Y_test, sample_size, test_dataset_ratio, MAX_DEPTH):
		CRITERION = "gini"
		RANDOM_STATE = 1
		tree_builder = DecisionTreeClassifier(criterion=CRITERION,
								              max_depth=MAX_DEPTH,
									          random_state=RANDOM_STATE)
		decision_tree = tree_builder.fit(X_train, Y_train)
		
		# plot classification result
		classifier = decision_tree
		classifier_name = "Decision Tree"
		label_names = self.__get_label_names("iris", remark="with_unit")
		x_label = label_names[0]
		y_label = label_names[1]
		scatter_name_dict = self.__get_classes("iris", "dict")
		plotting = Plotting(classifier,
						    classifier_name,
						    x_label,
							y_label,
							scatter_name_dict)
		X_combined = np.vstack((X_train, X_test))
		Y_combined = np.hstack((Y_train, Y_test))
		save_path = "res/decision_tree/sklearn_decision_tree/"+\
			        f"iris_decision_tree__decision_boundaries__[MAX_DEPTH={MAX_DEPTH}].png"
		test_idx = range(int(sample_size*(1-test_dataset_ratio)), sample_size)
		plotting.plot_classification(X_combined, Y_combined,
								     save_path, test_idx)
		return decision_tree
	
	def show_decision_tree_scikit(self, decision_tree, plot_type, MAX_DEPTH):
		if plot_type in ("Scikit-learn", "GraphViz"):
			print("[INFO] Generating decision tree plot ...")
			if plot_type == "Scikit-learn":
				# Obtain plot of decision tree by Scikit-learn
				sklearn_tree_model.plot_tree(decision_tree)
				save_path = "res/decision_tree/sklearn_decision_tree/"+\
					        f"iris_decision_tree__readable_decision_tree_(scikit)__MAX_DEPTH={MAX_DEPTH}.png"
				plt.savefig(save_path)
			else:
				# Obtain plot of decision tree by GraphViz
				save_path = "res/decision_tree/sklearn_decision_tree/"+\
					        f"iris_decision_tree__readable_decision_tree_(pydotplus)__MAX_DEPTH={MAX_DEPTH}.png"
				classes = self.__get_classes("iris", "list")
				label_names = self.__get_label_names("iris", remark="without_unit")
				dot_data = export_graphviz(decision_tree,
								           filled=True,
										   rounded=True,
										   class_names=classes,
										   feature_names=label_names,
										   out_file=None)
				graph = graph_from_dot_data(dot_data)
				graph.write_png(save_path)
		else:
			print("Cannot execute func: `show_decision_tree_scikit()`"+\
			      " because input parameter: `plot_type` is unknown.")

if __name__ == "__main__":
	decision_tree_generator = Decision_Tree_Generator()
	
	''' 1. Create & plot the "activity" decision tree '''
	print("[INFO] Decision Tree 1: \"activity\" decision tree ...\n")
	data, features = decision_tree_generator.load_data("activity")
	print("[INFO] Create decision tree ...\n")
	decision_tree = decision_tree_generator.create_tree(data, features)
	print("[INFO] Enumerate all possible combinations of decision tree ...\n")
	# ---
	combinations = decision_tree_generator.list_combinations(decision_tree)
	#decision_tree_generator.show_decision_tree(combinations)
	# ---
	options = [1,1,1,0]
	decision_tree_generator.predict(features, options, combinations)
	
	''' 2. Create & plot the "Iris" decision tree '''
	print("[INFO] Decision Tree 2: \"Iris\" decision tree ...\n")
	X, Y = decision_tree_generator.load_data("iris")
	sample_size = np.bincount(Y).sum()
	test_dataset_ratio = 0.3
	MAX_DEPTH = 8
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dataset_ratio, random_state=1, stratify=Y)
	decision_tree = decision_tree_generator.create_tree_scikit(X_train, X_test, Y_train, Y_test, sample_size, test_dataset_ratio, MAX_DEPTH)
	# ---
	#plot_type = "Scikit-learn"
	plot_type = "GraphViz"
	decision_tree_generator.show_decision_tree_scikit(decision_tree, plot_type, MAX_DEPTH)
	print(X)