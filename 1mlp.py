import numpy as np
from random import random

#DONE: save activations and derivatives
#DONE implement backpropagation
#DONE implement gradient descent
#DONE implement a trained method using backpropagation & gradient descent
#train our network with some dummy dataset
# make some predictions
 
class MLP:
	'''
	Multilayer Perceptron Class
	'''
	# constructor
	def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs= 2):
		#store the arguments internally
		self.num_inputs = num_inputs #num of neurons in input layer
		self.num_hidden = num_hidden #each elements represents the  
		# of neurons in each hidden layer of the neural network 
		self.num_outputs = num_outputs

		#list with number of neurons in each layer; indexed from  [0, num of layers)
		layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

		#initiate random weights for the layers
		weights = []
		#iterate through all layers and create a matrix weight for each pair of layers 
		for i in range(len(layers)-1):
			#generate the weight matrices with random values such that the matrix 
			#dimensions matching the # neurons in current layer + # of neurons in next layer
			w = np.random.rand(layers[i], layers[i+1])
			#append list with weight matrices as items for total layers - 1 as the max
			weights.append(w) 
		self.weights = weights

		
		activations = []
		for i in range(len(layers)):
			a = np.zeros(layers[i]) 
			activations.append(a) 
		self.activations = activations

		derivatives = []
		#derivatives are derivatives of E wrt weights; layers -1 
		for i in range(len(layers)-1):
			d = np.zeros((layers[i], layers[i+1]))
			derivatives.append(d) 
		self.derivatives = derivatives


	def forward_propagate(self, inputs):

		#for first layer, the activations are inputs
		activations = inputs
		self.activations[0] = inputs

		#move through each layer from left to right 
		for i, w in enumerate(self.weights):
			#calculate the net input
			net_inputs = np.dot(activations, w)

			#apply the sigmoid activation function
			activations = self._sigmoid(net_inputs)
			self.activations[i+1] = activations

		#Indexing explanation
		#a_3 = sigmoid(h_3) wherein,
		##h_3 is given by A_2 * W_2 i.e. from previous layer

		return activations


	def back_propagate(self, error, verbose=False):

		# dE/ dW_i = (y - a[i+1]) s'(h[i+1])) a_i --> iterable definition
		# s'(h_[i+1]) = s(h_[i+1]) (1- s(h_[i+1])) --> in terms of sigmoid
		# s(h_[i+1]) = a_[i+1] --> 

		#dE/ dW_[i-1] = (y - a[i+1]) s'(h[i+1])) W_i s'(h_i) a_[i-1] 

		#move from right to left from output to inputs; reverse index
		for i in reversed(range(len(self.derivatives))):
			activations = self.activations[i+1]
			delta = error * self._sigmoid_derivative(activations)
			current_activations = self.activations[i] #ndarray([0.1, 0.2]) --> vertical vector([0.1], [0.2])

			#need to rearrange arrays in vertical format for dot product to work correctly
			current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
			delta_reshaped = delta.reshape(delta.shape[0], -1).T

			self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
			
			error = np.dot(delta, self.weights[i].T)

			if verbose:
				print("Derivatives for W{}: {} ".format(i, self.derivatives[i]))

		return error


	def gradient_descent(self, learning_rate):
		for i in range(len(self.weights)):
			#retrieve weights and derivatives for the current layer
			weights = self.weights[i]
			#print("Original W{} {}".format(i,weights))

			derivatives= self.derivatives[i]

			#apply gradient descent 
			weights += derivatives * learning_rate
			#print("Updated W{} {}".format(i,weights))


	def train(self,inputs, targets, epochs, learning_rate):
		#num of epochs tells us how many times we want to feed the whole dataset. to the NN
		# higher num of epochs should improve the prediction accuracy
		#epochs is an integer
		for i in range(epochs):
			sum_error = 0

			#Now, go through all inputs and targets one by one, unpacking idx
			for input, target in zip(inputs, targets):

				#forward propagation
				output = self.forward_propagate(input)

				#calculate the error
				error = target - output

				#back_propagation
				self.back_propagate(error)

				#apply gradient descent
				self.gradient_descent(learning_rate)

				#this should hopefully decrease over iterations
				sum_error  += self._mse(target, output)

			#report error, normalized by inputs
			print("Error: {} at epoch {}".format(sum_error/ len(inputs), i))


	def _mse(self, target, output):
		return np.average((target-output)**2)


	def _sigmoid_derivative(self, x):
		return x*(1.0 - x)


	def _sigmoid(self,x):
		return 1.0 /(1.0 + np.exp(-x))


#main driver 
if __name__ == "__main__":


	#create a dataset to train a network for the sum operation
	inputs = np.array([[random()/2 for _ in range(2)] for 
		_ in range(1000)])  #array([[0.1, 0.2], [0.3, 0.4]])
	targets = np.array([[i[0] + i[1]] for i in inputs])  #array([[0.3], [0.7])
	#create an MLP using default values
	mlp = MLP(2,[5],1)

	#create some dummy inputs
	#input = np.array([0.1, 0.2])
	#target = np.array([0.3]) #sum of the above - let's see if n/w can learn sum operation
	#inputs = np.random.rand(mlp.num_inputs)

	#train our mlp
	mlp.train(inputs, targets, 50, 0.1)

	#forward propagation
	#outputs = mlp.forward_propagate(inputs)
	#output = mlp.forward_propagate(input) #rather the prediction

	#calculate the error
	#error = target - output

	#back_propagation
	#mlp.back_propagate(error)

	#apply gradient descent
	#mlp.gradient_descent(learning_rate=1)


	#print the results
	#print("The network intput is {}".format(inputs))
	#print("The network output is {}".format(outputs))
	#print("")




