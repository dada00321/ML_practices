import numpy as np

class Decision_Tree_Generator():
	def load_data(self):
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
		'''
		
		#data = [["no"],["yes"],["yes"],["no"],["yes"]]
		#data = [["no"],["yes"],["yes"],["no"],["no"]]
		
		features = ["天氣","溫度","濕度","風速"]
		return data, features
	
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
		
		''' Step 2. Find the "best feature" ("feature to split") within current data '''
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
		
		''' Step 3. Obtain the best feature '''
		# *** [TEST] ***
		best_feature = features[best_feature_idx]
		#print("[TEST] Feature to split:")
		#print(f"{best_feature} (idx: {best_feature_idx})\n")
		
		''' Step 4. Form up the decision tree '''
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
	
	def list_combinations(self, tree, prefix=None, combinations=[], delimiter=" => "):
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
						self.list_combinations(tmp, msg, combinations)
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
		else:
			print("[INFO] Check whether the data is incomplete.")
	
if __name__ == "__main__":
	decision_tree_generator = Decision_Tree_Generator()
	
	data, features = decision_tree_generator.load_data()
	print("[INFO] Create decision tree ...\n")
	decision_tree = decision_tree_generator.create_tree(data, features)
	
	print("[INFO] Enumerate all possible combinations of decision tree ...\n")
	combinations = decision_tree_generator.list_combinations(decision_tree)
	decision_tree_generator.show_decision_tree(combinations)
	
	options = [1,1,1,0]
	decision_tree_generator.predict(features, options, combinations)
	