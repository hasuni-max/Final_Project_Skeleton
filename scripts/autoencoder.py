import numpy as np
from processing import one_hot_encoding

class autoencode():
	"""
		Autoencode class! This class functions identically to the neural network class however unlike 
		the neural net, the dimensions of weights 1 and 2 in the autoencoder are tranposes of each other. 
		Therefore, the input and output are of the same size. 
	"""
	def __init__(self,data):
		self.data = data 
		np.random.seed(20)
		self.weights1 = np.random.rand(self.data.shape[1],3)
		self.weights2 = np.random.rand(3,self.data.shape[1])

	def embed(self):
		"""
			Reduce the size of input by doting to weights1
		"""
		self.hidden = self.sigmoid(np.dot(self.data, self.weights1))

	def decode(self):
		"""
			We can think of mutliplying back into the input dimensions as decoding here. We do this 
			by dotting with weghts2
		"""

		self.out = self.sigmoid(np.dot(self.hidden,self.weights2))

	def backpropogate(self):

		"""
			After each feedforward the derivatives are calculated and used to adjust weights1 and weights2
			The derivatives here calculated using the chain rule. Note that d_weights2 is caculated first
			and then d_weights1 follows. The derv of the MSE cost function is written here as 2*(self.data - self.out)

			The derivative of the sigmoid is provided in a seperate method. 
		"""
		
		#print(self.data - self.out)
		d_weights2 = np.dot(self.hidden.T, (2*(self.data - self.out) * self.sigmoid_derivative(self.out)))
		d_weights1 = np.dot(self.data.T,  (np.dot(2*(self.data - self.out) * self.sigmoid_derivative(self.out), self.weights2.T) * self.sigmoid_derivative(self.hidden)))
		
		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2


	def learn(self,epochs):
		"""
			Epochs is manually given for learning here. Not based on convergencec. For each epoch,
			we embed, then we decode and finally determine how different the input sequence is with the 
			one generated from the encoding. 
		"""

		for x in range(epochs):
			self.embed()
			self.decode()
			self.backpropogate()

		#print(np.sum((self.data-self.out)))

	def sigmoid(self,x):
		return 1.0/(1+ np.exp(-x))
	
	def sigmoid_derivative(self,x):
		return x * (1.0 - x)


if __name__ == "__main__":
	
	dna = ["AGATG","TATGT","ATATA","AGGGA"]
	one_hots = []
	for x in dna:
		one_hots.append(one_hot_encoding(x))

	one_hots = np.array(one_hots)
	# print(one_hots.shape)

	A = autoencode(one_hots)

	# print(A.weights)
	A.learn(1000)

	
