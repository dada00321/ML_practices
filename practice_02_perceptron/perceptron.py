import numpy as np

class Perceptron:
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
	RANDOM_STATE: int
	  random number seed (for generating initial weights)
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
			return True
		else:
			print("[WARNING] type of `Y` should be ndarray.")
			return False

	def check_parameters(self, LR, EPOCHS, RANDOM_STATE, THRESHOLD):
		if all([str(e).isdigit() for e in (LR, EPOCHS, RANDOM_STATE, THRESHOLD)]):
			return True
		else:
			if type(LR).__name__ == "float":
				return True
			print([str(e).isdigit() for e in (LR, EPOCHS, RANDOM_STATE, THRESHOLD)])
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

	def set_RANDOM_STATE(self, RANDOM_STATE):
		if type(RANDOM_STATE).__name__ == "int":
			self.RANDOM_STATE = RANDOM_STATE
			return True
		else:
			print("[WARNING] type of `RANDOM_STATE` should be integer.")
			return False

	def set_THRESHOLD(self, THRESHOLD):
		if type(THRESHOLD).__name__ in ("int", "float"):
			self.THRESHOLD = THRESHOLD
			return True
		else:
			print("[WARNING] type of `THRESHOLD` should be "+\
			      "integer or float.")
			return False

	def initialize_weights(self, X, random_state, threshold):
		"""
		This block must be late than block `set_X()` during execution.
		"""
		num_input_neurons = X.shape[1] # <= (1+m) nurons

		# type 1: set W to "small random values"
		random_generator = np.random.RandomState(random_state)
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
		"""
		dim. of W: 1+m
		dim. of X: 1+m
		rtn_val = w_1*x_1 + ... + w_m*x_m + b (b: bias, = threshold)
		"""
		return np.dot(W, X)

	def activation(self, z):
		"""
		"Predict" the exception outcome (y_i_hat)
		Depends on `THRESHOLD`
		"""
		return np.where(z>=0, 1, -1)

	def fit(self,
		    X, Y,
		    LR=1e-2,
			EPOCHS=50,
			RANDOM_STATE=1,
			THRESHOLD=0):
		self.errors_in_epochs = list()
		if all((self.check_parameters(LR, EPOCHS, RANDOM_STATE, THRESHOLD),
		        self.set_X(X),
				self.set_Y(Y),
				self.set_LR(LR),
				self.set_EPOCHS(EPOCHS),
				self.set_RANDOM_STATE(RANDOM_STATE),
				self.set_THRESHOLD(THRESHOLD))):
			print("[INFO] Input parameters are already received.")

			# Step 1: Initially, set weights to 0s or small random values
			self.initialize_weights(self.X, self.RANDOM_STATE, self.THRESHOLD)


			for _ in range(self.EPOCHS): # number of training iterations
				errors = 0
				for x_i, y_i in zip(self.X, self.Y): # i th sample
					# Step 2(a): Compute ŷ_i
					z = self.net_input(self.W, x_i) # both self.W and x_i have (1+m) neurons, start from w_0 anf x_0 respectively
					y_hat = self.activation(z) # binomial: +1 or -1

					# Step 2(b-1): Update weights except w_0
					'''
					for j in range(1, self.X.shape[1]):
						# select 1~"m" from [0~m] | self.X.shape[1]: (1+m)
						weight_j_increment = self.LR * (y_i - y_hat) * x_i[j]
						self.W[j] += weight_j_increment
					   ↓   ↓   ↓
					'''
					residual = y_i - y_hat
					errors += int(residual != 0)

					'''
					self.W[1:] += self.LR * residual * x_i[1:]

					# Step 2(b-2): Also adjust w_0 (variant of threshold/bias)
					self.W[0] += self.LR * residual
					'''
					self.W += self.LR * residual * x_i
				self.errors_in_epochs.append(errors)
			self.error_times = list()
			return self.X

		else:
			print("[WARNING] `Perceptron` can not be initialized, some errors occur!")
			self.is_available = False
			return None