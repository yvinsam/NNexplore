import json
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "data.json"

#load data
def load_data(data_path):
	''' Loads training dataset from json file

	:param data_path (str): Path to json file containing data
	:return X (ndarray): Inputs
	: retrun y (ndarray): Targets
	'''
	with open(data_path, "r") as fp:
		data = json.load(fp)

	#convert lists into numpy arrays
	X = np.array(data["mfcc"])
	y = np.array(data["labels"])

	return X, y

def prepare_datasets(test_size, validation_size):
	
	#load data
	X, y = load_data(DATA_PATH)
	#create the train/test split
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)	
	# create train/validation split
	X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train, test_size=validation_size)	

	#3d array input --> (130 time bins, 13 mfcc values, 1] format
	X_train  = X_train[..., np.newaxis] #4D array --> [num_samples, 130, 13, 1] take what I have as ..., and give me another new axis
	X_validation  = X_validation[..., np.newaxis]
	X_test  = X_test[..., np.newaxis]

	return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

	#create model; cnn with 3 layers
	model = keras.Sequential()
	
	# 1st Conv layer + max pooling layer
	model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
	model.add(keras.layers.MaxPool2D((3,3), strides=(2,2),padding='same'))
	model.add(keras.layers.BatchNormalization())

	# 2nd Conv layer
	model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
	model.add(keras.layers.MaxPool2D((3,3), strides=(2,2),padding='same'))
	model.add(keras.layers.BatchNormalization())

	# 3rd Conv layer
	model.add(keras.layers.Conv2D(32, (2,2), activation='relu'))
	model.add(keras.layers.MaxPool2D((2,2), strides=(2,2),padding='same'))
	model.add(keras.layers.BatchNormalization())

	# flatten the output and feed the 1D array into dense layer
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dropout(0.3)) #to handle overfitting

	# output layer 
	model.add(keras.layers.Dense(10, activation='softmax')) #neurons = # classes to predict

	return model

def predict(model, X,y):

	X = X[np.newaxis,...] #insert a new axis at beginning; 4D needed
	
	#prediction 2D = [[0.1, 0.2, 0,...]]
	prediction = model.predict(X) #X -> [1, 130, 13, 1]

	#extract index with max value
	predicted_index = np.argmax(prediction,axis=1) #[3]
	print("Expected index: {}, Predicted index:{}".format(y,predicted_index))

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
	
	# create train, validation, and test sets
	X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(test_size=0.25, validation_size=0.2)

	# build the CNN net
	input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
	model = build_model(input_shape)

	# compile the network
	optimizer = keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=optimizer,
		loss="sparse_categorical_crossentropy",
		metrics=['accuracy'])

	#model.summary()

	# train the network
	history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

	#plot accuracy/error for training and validatino
	plot_history(history)

	#evaluate CNN on the test set
	test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
	print("Accuracy on test set is {}".format(test_accuracy))


	#run inference: make predictions on a sample
	X = X_test[50]
	y = y_test[50]

	predict(model, X, y)














