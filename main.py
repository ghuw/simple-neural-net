import numpy as np
import dill

#Extract and prepare MNIST training data
def LoadMNISTSets(fileimage, n):
	fimg = open(fileimage, "rb")
	
	fimg.read(16) # 16 where MNIST's image pixel started.
	number_of_pix = 28*28
	s = (n,number_of_pix)
	image_sets = np.zeros(s)

	for i in range(n):
		for j in range(number_of_pix):
			pix = ord(fimg.read(1))
			image_sets[i,j] = normalize(pix)
			
	fimg.close()
	return image_sets

#Extract and prepare MNIST test data
def LoadMNISTLabel(filelabel, n):
	flabel = open(filelabel, "rb")
	
	flabel.read(8) # 8 where MNIST's image label started.
	s = (n,10)
	label_sets = np.zeros(s, dtype=int)
	
	for i in range(n):
		label = ord(flabel.read(1))
		label_sets[i,label] = 1

	flabel.close()
	return label_sets

#Normalization from [0..255] to [0..1] scale
def normalize(dataset):
	return dataset / 255
	
#Assign initial weights and bias to each node in the layer
class LayerNode():
	def __init__(self, number_of_nodes, number_of_inputs):
		self.number_of_nodes = number_of_nodes
		self.number_of_inputs = number_of_inputs
		self.weights = np.random.normal(0, 0.001, size=(number_of_inputs, number_of_nodes)) #initial weights using normal distribution
		self.bias = np.random.normal(0, 0.001, size=(1, number_of_nodes))

class NeuralNetwork():
	def __init__(self, layer1, layer2, layer3):
		self.layer1 = layer1
		self.layer2 = layer2
		self.layer3 = layer3
		
	# Softmax activation function
	def __softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / np.sum(e_x)
	
	# ReLU activation function
	def __relu(self, x):
		return x * (x > 0)
	
	# Tanh activation function
	def tanh(self, layer):
        return np.tanh(layer)	
	
	def calculate_error(self, labels, layer_output):
		# Cross Entropy loss
		return np.negative(np.sum(np.multiply(labels, np.log(layer_output))))
	
	def forward_pass(self, inputs):
		#Sum of dot product of each layer
		#Apply activation function to the layer sums		
		
		#Pass the training sets through neural network
		z_layer_1 = np.dot(inputs, self.layer1.weights) + self.layer1.bias
		output_layer_1 = self.__relu(z_layer_1) #activation(1)
		
		z_layer_2 = np.dot(output_layer_1, self.layer2.weights) + self.layer2.bias
		output_layer_2 = self.__relu(z_layer_2) #activation(2)
		
		z_layer_3 = np.dot(output_layer_2, self.layer3.weights) + self.layer3.bias #z4
		output_layer_3 = self.__softmax(z_layer_3) #y_Hat
		
		return output_layer_1, output_layer_2, output_layer_3
	
	def backward_pass(self, learning_rate, labels, training_inputs, output_layer_1, output_layer_2, output_layer_3):
		targets = labels
		delta_layer3 = output_layer_3 - targets
		delta_layer2 = (delta_layer3).dot(self.layer3.weights.T) * output_layer_2 * (1 - output_layer_2)
		delta_layer1 = (delta_layer2).dot(self.layer2.weights.T) * output_layer_1 * (1 - output_layer_1)
		
		#Adjust bias and weights
		self.layer3.weights -= learning_rate * output_layer_2.T.dot(delta_layer3)
		self.layer3.bias -= learning_rate * (delta_layer3).sum(axis=0)
		
		self.layer2.weights -= learning_rate * output_layer_1.T.dot(delta_layer2)
		self.layer2.bias -= learning_rate * (delta_layer2).sum(axis=0)
		
		self.layer1.weights -= learning_rate * training_inputs.T.dot(delta_layer1)
		self.layer1.bias -= learning_rate * (delta_layer1).sum(axis=0)
		
	def train(self, batch_size, training_inputs, labels, n_epochs, learning_rate, filename):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		for j in range(n_epochs):
			i = 0
			print("/// EPOCH: ", j+1, "/", n_epochs, " ///")
			
			output_layer_1, output_layer_2, output_layer_3 = self.forward_pass(training_inputs)
			layer3_error = self.calculate_error(labels, output_layer_3)
			self.backward_pass(learning_rate, labels, training_inputs, output_layer_1, output_layer_2, output_layer_3)
			
			error_result = layer3_error / len(output_layer_3)
			print("n\Error: ", error_result)
			
		print("Saving...")
		dill.dump_session(filename)	
	
	def test_accuracy(self, test_inputs, test_outputs):
		output_layer_1, output_layer_2, output_layer_3 = self.forward_pass(test_inputs)
		number_of_instance = len(test_outputs)
		nearest_node = np.argmax(output_layer_3, axis=1)
		test_real_value = np.argmax(test_outputs, axis=1)
		correct = (nearest_node == test_real_value).sum()
		print ("Accuracy: ", correct*100/ number_of_instance,"%")
			
if __name__ == "__main__":
		
		print("Starting...")
		
		print("Training...")
		#Seed random generator
		np.random.seed(1)
		
		#Initialize number of nodes of each layer
		number_of_inputs_node = 28*28
		number_of_layer1_node = 10
		number_of_layer2_node = 4
		number_of_outputs_node = 10
		
		#Create layers of neural network
		layer1 = LayerNode(number_of_layer1_node, number_of_inputs_node)
		layer2 = LayerNode(number_of_layer2_node, number_of_layer1_node)
		layer3 = LayerNode(number_of_outputs_node, number_of_layer2_node)
		
		#Combine each layer into a neural network
		neural_network = NeuralNetwork(layer1,layer2,layer3)
		
		#Define number of training data 
		n_training = 60000
		#Load training sets
		training_inputs = LoadMNISTSets("train-images.idx3-ubyte", n_training)
		training_outputs = LoadMNISTLabel("train-labels.idx1-ubyte", n_training)
		
		#Train the network using training sets
		neural_network.train(1, training_inputs, training_outputs, n_epochs=5, learning_rate=0.001, filename="model.pkl")
		
		print("Testing...")
		#Define number of test data
		n_test = 10000
		#Load test data
		test_inputs = LoadMNISTSets("t10k-images.idx3-ubyte", n_test)
		test_outputs = LoadMNISTLabel("t10k-labels.idx1-ubyte", n_test)
		
		#Test and check accuracy of the neural network
		neural_network.test_accuracy(test_inputs, test_outputs)