#implementing a simple artificial neuron using sum and activation fns
import math

def sigmoid(x):
	y  = 1.0 /(1.0 + math.exp(-x))
	return y

def activate (inputs, weights):
	h= 0
	#calculate the net input sum first
	for x, w in zip(inputs, weights):
		h += x*w

	#calculate the activation; sigmoid here
	return sigmoid(h)

#main driver
if __name__ == '__main__':

	inputs = [.5, .3, .2]
	weights = [.4, .7, .2]
	output = activate(inputs,weights)
	print('Here {}'.format(output)) 




  