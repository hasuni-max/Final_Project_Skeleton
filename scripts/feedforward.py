import numpy as np

#https://stats.stackexchange.com/questions/124651/how-do-i-incorporate-the-biases-in-my-feed-forward-neural-network

class NeuralNetwork:
	def __init__(self, x, y,hidden_layer_size=3,lr=.1,bias=True):
		self.input = x
		self.hidden_layer_size = hidden_layer_size
		self.weights1 = np.random.rand(self.input.shape[1],self.hidden_layer_size) 
		self.weights2 = np.random.rand(self.hidden_layer_size,1)

		self.lr = lr #incorporate this  

		self.y = y
		self.output = np.zeros(self.y.shape)

	def feedforward(self):

		#original
		self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
		self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
	
	def backprop(self):
		# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

		#Original
		d_weights2 = self.lr * np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
		d_weights1 = self.lr * np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))
		
		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def predict(self,test):
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




