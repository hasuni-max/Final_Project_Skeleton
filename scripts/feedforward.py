import numpy as np


class NeuralNetwork:
	def __init__(self, x, y,hidden_layer_size=3,lr=.1,bias=True):
		self.input = x
		self.hidden_layer_size = hidden_layer_size
		self.weights1 = np.random.rand(self.input.shape[1],self.hidden_layer_size) 
		self.weights2 = np.random.rand(self.hidden_layer_size,1)

		#Learning rate: This value is multiplied to the gradient in the backprop method
		self.lr = lr 

		self.y = y
		self.output = np.zeros(self.y.shape)

	def feedforward(self):

		"""
			Layer1 consists of taking the dot between the input layer and weights1. Sigmoid is applied afterwards
			Layer2: The output from layer1 is then dotted with weights2. Sigmoid is applied afterward
		"""

		self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
		self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
	
	def backprop(self):
		"""
			After each feedforward the derivatives are calculated and used to adjust weights1 and weights2
			The derivatives here calculated using the chain rule. Note that d_weights2 is caculated first
			and then d_weights1 follows. The derv of the MSE cost function is written here as 2*(self.y - self.output)

			The derivative of the sigmoid is provided in a seperate method. 
		"""

		d_weights2 = self.lr * np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
		d_weights1 = self.lr * np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))
		
		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def predict(self,test):
		"""
			This method is used to predict new inputs by multiplying the input vector with weights1 and 
			then weights2. Sigmoid is applied to both calculations.
		"""

		predict1 = self.sigmoid(np.dot(test, self.weights1))
		predict2 = self.sigmoid(np.dot(predict1, self.weights2))

		return predict2

	def sigmoid(self,x):
		return 1.0/(1+ np.exp(-x))

	def sigmoid_derivative(self,x):
		return x * (1.0 - x)




if __name__ == "__main__":


	X = np.array([[0,0,1,4,3,1],
                  [0,1,1,3,4,5],
                  [1,0,1,1,1,1],
                  [1,1,1,2,3,4]])
	y = np.array([[0],[1],[1],[0]])
	nn = NeuralNetwork(X,y)

	for i in range(1500):
		nn.feedforward()
		nn.backprop()


	print(nn.predict(np.array([0,1,1,4,5,3])))
	#print(np.dot(nn.output.T,np.array([0,1,1])))




