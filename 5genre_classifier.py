'''
LOAD DATA, Train a NN, check for overfitting

@nsamudrala, Oct 2020

'''
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from matplotlib import pyplot as plt

DATASET_PATH = "data.json"

#load data
def load_data(dataset_path):
	with open(dataset_path, "r") as fp:
		data = json.load(fp)

	#convert lists into numpy arrays
	inputs = np.array(data["mfcc"])
	targets = np.array(data["labels"])

	return inputs, targets

def plot_history(history):

	fig, axs = plt.subplots(2)

	#create accuracy subplots
	axs[0].plot(history.history["accuracy"], label="train_accuracy")
	axs[0].plot(history.history["val_accuracy"], label="test accuracy")
	axs[0].set_ylabel("Accuracy")
	axs[0].legend(loc="lower right")
	axs[0].set_title("Accuracy eval")
 
	#create error subplots
	axs[1].plot(history.history["loss"], label="train_error")
	axs[1].plot(history.history["val_loss"], label="test error")
	axs[1].set_ylabel("Error")
	axs[1].set_xlabel("Epoch")
	axs[1].legend(loc="upper right")
	axs[1].set_title("Error eval")

	plt.show()



if __name__ == '__main__':

	#load data
	inputs, targets = load_data(DATASET_PATH)

	#split the data into train and test sets
	inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

	#build the network architecture 1--> 3 --> 1
	model = keras.Sequential([

		#input layer is a 3D array; first one is segment_index; data in other 2
		keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

		#lst hidden layers
		keras.layers.Dense(512, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
		#relu: rectified linear unit; has better convergence+ reduces likelihood of vanishing gradient
		keras.layers.Dropout(0.3),

		#lst hidden layers
		keras.layers.Dense(256, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
		keras.layers.Dropout(0.3),

		#lst hidden layers
		keras.layers.Dense(64, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
		keras.layers.Dropout(0.3),

		#output layer
		keras.layers.Dense(10,activation="softmax")

		])

	#compile network
	optimizer = keras.optimizers.Adam(learning_rate=0.0001)
	#Adam is a variation of Stochastic Gradient Descent/ works well for DL
	model.compile(optimizer = optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	model.summary()


	#train network
	history = model.fit(inputs_train, targets_train, 
		validation_data=(inputs_test, targets_test),
		epochs=50,
		batch_size=32) #batching is how we train the n/w or back-propagate; stochastic/quick/1-sample; full batch/slow/intensive; mini-batch: 16 - 128

	#plot accuracy and error over the epochs
	plot_history(history)





























