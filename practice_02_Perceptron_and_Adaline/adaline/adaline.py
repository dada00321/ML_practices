import numpy as np

class Adaline:
	"""
	Parameters
	------------
	X: ndarray (numpy.ndarray)
	  Inputs
    Y: ndarray (numpy.ndarray)
	  Real labels (classes)
    W: ndarray (numpy.ndarray) / 1d-array
	  Weights
	LR: float
	  learning rate (η, between 0.0 and 1.0)
	EPOCHS: int
	  # of training iterations (>= 1)
	RANDOM_SEED: int
	  random number seed for generating initial weights
    THRESHOLD: int or float
	  a critical value, be used to compare with net input

	Attribures
	------------
	num_input_features: int
	  number of input features
	errors: array-like
	  difference between real label (class) and predicted label (class)

	"""
	def set_X(self, X):
		if type(X).__name__ == "ndarray":
			# For each sample, set the first input neuron x_0 to "1".
			X = np.insert(X, 0, 1, axis=1)
			# After insertion, X have (1+m) input neurons.
			self.X = X
			return True
		else:
			print("[WARNING] type of `X` should be ndarray.")
			return False

	def set_Y(self, Y):
		if type(Y).__name__ == "ndarray":
			self.Y = Y
			#print(Y)
			return True
		else:
			print("[WARNING] type of `Y` should be ndarray.")
			return False

	def check_parameters(self, LR, EPOCHS, RANDOM_SEED, THRESHOLD):
		if all([str(e).isdigit() for e in (LR, EPOCHS, RANDOM_SEED, THRESHOLD)]):
			return True
		else:
			if type(LR).__name__ == "float":
				return True
			print([str(e).isdigit() for e in (LR, EPOCHS, RANDOM_SEED, THRESHOLD)])
			print("[WARNING] Learning rate, EPOCHS and RANDOM_INIT_NUM\n"+\
		          " must be numerics.")
			return False

	def set_LR(self, LR):
		if 0.0 <= LR <= 1.0:
			if type(LR).__name__ == "float":
				self.LR = LR
				return True
			else:
				print("[WARNING] type of `LR` should be float.")
				return False
		else:
			print("[WARNING] `LR` must greater than "+\
				  "or equal to 0.0, less than or equal to 1.0.")
			return False

	def set_EPOCHS(self, EPOCHS):
		if EPOCHS >= 1:
			if type(EPOCHS).__name__ == "int":
				self.EPOCHS = EPOCHS
				return True
			else:
				print("[WARNING] type of `EPOCHS` should be integer.")
				return False
		else:
			print("[WARNING] `EPOCHS` must greater than "+\
				  "or equal to 1.")
			return False

	def set_RANDOM_SEED(self, RANDOM_SEED):
		if type(RANDOM_SEED).__name__ == "int":
			self.RANDOM_SEED = RANDOM_SEED
			return True
		else:
			print("[WARNING] type of `RANDOM_SEED` should be integer.")
			return False

	def set_THRESHOLD(self, THRESHOLD):
		if type(THRESHOLD).__name__ in ("int", "float"):
			self.THRESHOLD = THRESHOLD
			return True
		else:
			print("[WARNING] type of `THRESHOLD` should be "+\
			      "integer or float.")
			return False

	def initialize_weights(self, X, RANDOM_SEED, threshold):
		"""
		This block must be late than block `set_X()` during execution.
		"""
		num_input_neurons = X.shape[1] # <= (1+m) nurons

		# type 1: set W to "small random values"
		random_generator = np.random.RandomState(RANDOM_SEED)
		W = random_generator.normal(loc=0.0, scale=0.01,
							        size=num_input_neurons)
		# type 2: set W to "0"s
		'''
		W = np.zeros(num_input_neurons)
		'''

		# For each sample, set the first weight w_0 to "-θ".
		W[0] = threshold*(-1)
		self.W = W

	def net_input(self, W, X):
		Z = np.dot(X, W)
		print(f"dim(X): {X.shape[0]} x {X.shape[1]}")
		print(f"dim(W): {W.shape} = 1 x {W.shape[0]}")
		print(f"dim(X•W): = dim(W) x dim(X.T) = (1 x {W.shape[0]}) x ({X.shape[1]} x {X.shape[0]})\n"+\
			  f"\t\t  = dim(Z) = {Z.shape} = 1 x {Z.shape[0]}")
		return Z

	def predict(self, linear_activation_output):
		return np.where(linear_activation_output >= 0, 1, -1)

	def fit(self,
		    X, Y,
		    LR=1e-2,
			EPOCHS=50,
			RANDOM_SEED=1,
			THRESHOLD=0):
		if all((self.check_parameters(LR, EPOCHS, RANDOM_SEED, THRESHOLD),
		        self.set_X(X),
				self.set_Y(Y),
				self.set_LR(LR),
				self.set_EPOCHS(EPOCHS),
				self.set_RANDOM_SEED(RANDOM_SEED),
				self.set_THRESHOLD(THRESHOLD))):
			#print("[INFO] Input parameters are already received.")

			# Step 1: Initially, set weights to 0s or small random values
			self.initialize_weights(self.X, self.RANDOM_SEED, self.THRESHOLD)
			self.costs = list()
			for _ in range(self.EPOCHS): # number of training iterations
				# Step 2: Update weights
				Z = self.net_input(self.W, self.X) # both self.W and x_i have (1+m) neurons, start from w_0 anf x_0 respectively
				'''
				# N: N training samples
				# X: [1, ..., 1] (for W_0: [w_0[1], ..., w_0[N]]) is an unit vector with 1s
				# Because it's unit vector, the author of ML book didn't take
				# X into account in weights updating with W_0 (i.e. W[0])

				self.W[0] += self.LR * (self.Y - Z).sum()
				self.W[1:] += self.LR * X.T.dot(self.Y - Z)
				'''
				# N: # of training samples
				# m: # of input neurons
				# X.T: transpose of matrix X
				# LR: learning rate, is a scalar
				#
				# dim(W): 1 x (1+m) // W.shape: ((1+m),)
				# dim(X): N x (1+m) // => X.shape: (N, (1+m))
				# dim(Z): 1 x N // dim(X•W) = dim(W) x dim(X.T)
				#
				# dim(Y): 1 x N
				# dim(Y-Z): 1 x N
				# LR * dim(Y-Z): 1 x N
				#
				# // dim(X.T • (Y-Z)) = dim(Y-Z) x dim(X)
				#
				# => wants dim(ΔW) = dim(W) = 1 x (1+m)
				'''
				self.W += self.LR * X.T.dot(self.Y - Z)
				'''
				errors = self.Y - Z
				#print("dim(errors):", errors.shape)
				# self.W += self.LR * np.dot(errors, X)
				W_increment = self.LR * np.dot(self.X.T, (self.Y - Z))
				#print(tmp.shape)
				self.W += W_increment
				cost = (errors**2).sum() / 2
				# Define: cost = 1/2 SSE
				self.costs.append(cost)
				print(f"dim(Y): {self.Y.shape} = 1 x {Y.shape[0]}")
				print(f"dim(Z): {Z.shape} = 1 x {Y.shape[0]}")
				print(f"dim(Y-Z): {errors.shape} = 1 x {errors.shape[0]}")
				print(f"dim(ΔW): {W_increment.shape} = 1 x {W_increment.shape[0]}")
			print("\n\n")
			return self.X

		else:
			print("[WARNING] `Adaline` can not be initialized, some errors occur!")
			self.is_available = False
			return None