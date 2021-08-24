from math import log

def cal_information_gain(tree, impurity_type, case_name):
	# formula: Information Gain
	# => IG = I(D_p) - summation_i( I(D_i) * (N_i/N_p) )
	IG = lambda I_p, I_children, N_p, N_children: I_p - sum([I_i * N_i / N_p for I_i, N_i in zip(I_children,N_children)]) 
	
	if impurity_type in ("gini", "entropy", "classification_error"):
		if impurity_type == "gini":
			# formula: Gini Impurity
			# => I_G = 1 - summation_i( p(i|t)^2 )
			I = lambda probs: 1 - sum([prob**2 for prob in probs])
		elif impurity_type == "entropy":
			# formula: Entropy (as the impurity)
			# => I_H = - summation_i( p(i|t) * log_2(p(i|t)) )
			I = lambda probs: - sum([prob * log(prob, 2) if prob != 0 else 0 for prob in probs])
		else:
			# formula: Classification error (as the impurity)
			# => I_E = 1 - max{ p(i|t) }
			I = lambda probs: 1 - max(probs)
	
	for D_p, D_children in tree.items():
		#print("D_p:", D_p)
		#print("D_children:", D_children)
		
		'''
		# 1. Calculate for basic info: probilities & smaple sizes
		print("list of prob(D_p):", 
		      [e/sum(D_p) for e in D_p])
		print("list of prob(D_children):", 
		      [[e/sum(kids) for e in kids]
		        for kids in D_children])
		
		print("num of D_p:", sum(D_p))
		print("nums of D_children:", 
		      [sum(kids) for kids in D_children])
		'''
		
		'''
		# 2. Get impurity of parent node & children nodes
		print(f"I_G(D_p) = {I_G([e/sum(D_p) for e in D_p])}")
		print("I_G(D_children) = "+\
		     f"{[I_G([e/sum(kids) for e in kids]) for kids in D_children]}")
		'''
		
		# 3. Calculate "Information Gain"
		I_p = I([e/sum(D_p) for e in D_p])
		I_children = [I([e/sum(kids) for e in kids]) for kids in D_children]
		N_p = sum(D_p)
		N_children = [sum(kids) for kids in D_children]
		information_gain = IG(I_p, I_children, N_p, N_children)
		print(f"[impurity measure: `{impurity_type}`]")
		floting_points = 4
		print(f"Information Gain of {case_name}: {round(information_gain, floting_points)}\n")
		return information_gain

if __name__ == "__main__":
	D_p = (40,40)
	D_left = (30,10)
	D_right = (10,30)
	tree_A = {D_p: [D_left, D_right]}
	
	D_p = (40,40)
	D_left = (20,40)
	D_right = (20,0)
	tree_B = {D_p: [D_left, D_right]}
	
	#[TEST] Show the result of each Information Gain
	#       using different impurity measure for case A & B
	cal_information_gain(tree_A, "gini", "case A")
	cal_information_gain(tree_B, "gini", "case B")
	cal_information_gain(tree_A, "entropy", "case A")
	cal_information_gain(tree_B, "entropy", "case B")
	cal_information_gain(tree_A, "classification_error", "case A")
	cal_information_gain(tree_B, "classification_error", "case B")
	