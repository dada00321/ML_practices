import numpy as np

class Adaline_SGD:
	"""
	Parameters
	------------
	X: ndarray (numpy.ndarray)
	  Inputs
    Y: ndarray (numpy.ndarray)
	  Real labels (classes)
    W: ndarray (numpy.ndarray) / 1d-array
	  Weights
    N:
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
	def set_X(self, X, is_first_time=True):
		if type(X).__name__ == "ndarray":
			if is_first_time:
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

	def shuffle_data(self, X, Y):
		random_val = self.random_generator.permutation(len(Y))
		return X[random_val], Y[random_val]

	def reset_X_and_Y(self, X, Y):
		self.set_X(X, False)
		self.set_Y(Y)

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

	def __initialize_weights(self, X, RANDOM_SEED, threshold):
		"""
		This block must be late than block `set_X()` during execution.
		"""
		num_input_neurons = X.shape[1] # <= (1+m) nurons

		# type 1: set W to "small random values"
		random_generator = np.random.RandomState(RANDOM_SEED)
		W = random_generator.normal(loc=0.0, scale=0.01,
							        size=num_input_neurons)
		self.random_generator = random_generator
		# type 2: set W to "0"s
		'''
		W = np.zeros(num_input_neurons)
		'''

		# For each sample, set the first weight w_0 to "-θ".
		W[0] = threshold*(-1)
		self.W = W

	def net_input(self, W, x_i):
		Z = np.dot(x_i, W)
		# dim(Z) = dim(W) x dim(X.T) = (1x(m+1)) x ((m+1)x1) = 1x1
		'''
		print(f"dim(X): {X.shape[0]} x {X.shape[1]}")
		print(f"dim(W): {W.shape} = 1 x {W.shape[0]}")
		print(f"dim(X•W): = dim(W) x dim(X.T) = (1 x {W.shape[0]}) x ({X.shape[1]} x {X.shape[0]})\n"+\
			  f"\t\t  = dim(Z) = {Z.shape} = 1 x {Z.shape[0]}")
		'''
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
			self.__initialize_weights(self.X, self.RANDOM_SEED, self.THRESHOLD)
			self.costs = list()
			for _ in range(self.EPOCHS): # number of training iterations
				tmpX, tmpY = self.shuffle_data(self.X, self.Y)
				self.reset_X_and_Y(tmpX, tmpY)
				sum_of_weight_increments = 0
				for x_i, y_i in zip(self.X, self.Y): # i th sample
					z_i = self.net_input(self.W, x_i) # dim(Z) = 1x1
					error = y_i - z_i
					w_i_increment = self.LR * x_i.dot(error)  # dim(err) = 1x1, dim(x_i) = 1x(m+1)
					self.W += w_i_increment # dim(W) = dim(w_i_increment) = 1 x (m+1)
					sum_of_weight_increments += (error/2)**2
				self.costs.append(sum_of_weight_increments / Y.shape[0])
			return self.X
		else:
			print("[WARNING] `Adaline` can not be initialized, some errors occur!")
			self.is_available = False
			return None