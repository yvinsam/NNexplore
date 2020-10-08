import numpy as np

#save activations and derivatives
#implement backpropagation
#implement gradient descent
#implement train using backpropagation & gradient descent
#train our net with some dummy dataset
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
		#derivatives are derivatives of E wrt weights; so layers - 1 
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

		#a_3 = s(h_3)
		#h_3 = A_2 * W_2

		return activations

	def _sigmoid(self,x):
		y  = 1.0 /(1.0 + np.exp(-x))
		return y

#main driver 
if __name__ == "__main__":


	#create an MLP using default values
	mlp = MLP()

	#create some inputs
	inputs = np.random.rand(mlp.num_inputs)

	#perform forward propagation
	outputs = mlp.forward_propagate(inputs)

	#print the results
	print("The network intput is {}".format(inputs))
	print("The network output is {}".format(outputs))




