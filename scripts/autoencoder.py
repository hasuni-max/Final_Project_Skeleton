import numpy as np
from processing import one_hot_encoding

class autoencode():
	"""
		Autoencode class! 
	"""
	def __init__(self,data):
		self.data = data 
		np.random.seed(20)
		self.weights1 = np.random.rand(self.data.shape[1],3)
		self.weights2 = np.random.rand(3,self.data.shape[1]) #consider using the transpose of weights1

	def embed(self):
		self.hidden = self.sigmoid(np.dot(self.data, self.weights1))

	def decode(self):
		self.out = self.sigmoid(np.dot(self.hidden,self.weights2))

	def backpropogate(self):
		
		#print(self.data - self.out)
		d_weights2 = np.dot(self.hidden.T, (2*(self.data - self.out) * self.sigmoid_derivative(self.out)))
		d_weights1 = np.dot(self.data.T,  (np.dot(2*(self.data - self.out) * self.sigmoid_derivative(self.out), self.weights2.T) * self.sigmoid_derivative(self.hidden)))
		
		# update the weights with the derivative (slope) of the loss function
		self.weights1 += d_weights1
		self.weights2 += d_weights2


	def learn(self,epochs):

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

	
