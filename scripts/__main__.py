import numpy as np
import sys
from processing import one_hot_encoding, parse_fasta, decoder, hamming_distance, generate_negative_samples
from autoencoder import autoencode
from feedforward import NeuralNetwork
from k_validation import k_cross_validation
from ROC import roc


def calculate_tp_fp(pos_matches,neg_matches,threshold):
	"""
		Calculate true positives and false positives given positive and negative
		matches, their corresponding sequences (sequences), a gap opening and a 
		gap extension penalty and finally a scoring matrix

	"""
	true_pos = 0
	false_neg = 0 
	true_neg = 0
	false_pos = 0 

	for pos in pos_matches:
		if pos > threshold:
			true_pos += 1
		else:
			false_neg += 1	
			
	for neg in neg_matches:

		if neg < threshold:
			true_neg += 1
		else:
			false_pos += 1	

	tp = true_pos/(true_pos+false_neg)
	fp = false_pos/(true_neg+false_pos)	

	return tp,fp


dna = []
with open("./data/rap1-lieb-positives.txt") as fh:
	for line in fh:
		line = line.strip()
		dna.append(line.upper())


if sys.argv[1] == "A":

	####################### Autoencoder ########################
	one_hots = []

	for x in dna:
		one_hots.append(one_hot_encoding(x[0:14]))

	one_hots = np.array(one_hots)

	A = autoencode(one_hots)
	A.learn(1000)

	#calculate avergae hamming distances between natural sequences and encoded sequences.
	#this gives you a general idea of well the autoencoder works. Also visually nice
	dists = []
	for i,x in enumerate(A.out):
		B = x.reshape(14,5)
		dist = hamming_distance(dna[i],decoder(B))
		dists.append(dist)

	print("Avg hamming distance between sequences and their respective encodings")
	print("Worst case scenario is a distance of 14, which would be the reverse sequence in this context")
	print(sum(dists)/len(dists))

	#SHOW AN EXAMPLE OF THE ENCODING
	print("Example sequence and its encoded self")
	print(dna[0][0:14])
	print(decoder(A.out[0].reshape(14,5)))


elif sys.argv[1] == "C":
	#######  try cross validation   #########

	pos_predictions = []
	neg_predictions = []

	neg = []
	for seq in generate_negative_samples("./data/yeast-upstream-1k-negative.fa",17,137,dna):
		neg.append(seq)

	#Using the k_cross_validation generator. 20-fold cross is used here
	for train_pos, train_neg, test_on in k_cross_validation(dna, neg, 20):

		Y = [np.array([1])]*len(train_pos) + [np.array([0])]*len(train_neg) #generate labels for current fold

		one_hot_pos = [one_hot_encoding(x) for x in train_pos] #list of positive one_hots. Note one_hot_encoding() returns numpy array
		one_hot_neg = [one_hot_encoding(x) for x in train_neg] #list of negative one_hots
		one_hots = one_hot_pos + one_hot_neg #combined lists

		X = np.array(one_hots) #convert to numpy array
		Y = np.array(Y) #again convert to numpy array

		nn = NeuralNetwork(X,Y,hidden_layer_size=2) #new neural net

		for i in range(1000):#train this fold
			nn.feedforward()
			nn.backprop()


		for positive, negative in test_on: #This is the test set from this fold. 
			#save into two lists. Predictions for each test fold are added to these lists

			pos_predictions.append(nn.predict(one_hot_encoding(positive)))
			neg_predictions.append(nn.predict(one_hot_encoding(negative)))


	#Generate roc curve 
	curvy = roc()
	for thresh in np.arange(0,1,.001): #Calculate tp and fp for several thresholds 

		for i in range(len(pos_predictions)):
			tp,fp = calculate_tp_fp(pos_predictions,neg_predictions,thresh)
			curvy.add_rates(tp,fp)

	curvy.plot_ROC(lab="Standard NeuralNetwork")
	curvy.show_plot()


elif sys.argv[1] == "T":

	################### Generate test predictions ################

	
	#Test predictions are generated using the entire training set!
	

	test=[]
	with open("./data/rap1-lieb-test.txt") as fh:
		for line in fh:
			line = line.strip()
			test.append(line.upper())

	#Train on whole training 

	one_hots = []
	Y = []
	for x in dna:
		one_hots.append(one_hot_encoding(x))
		Y.append(np.array([1]))

	neg = []
	for seq in generate_negative_samples("./data/yeast-upstream-1k-negative.fa",17,137,dna):
		neg.append(seq)
		one_hots.append(one_hot_encoding(seq))
		Y.append(np.array([0]))

	Y = np.array(Y)
	X = np.array(one_hots)

	nn = NeuralNetwork(X,Y)

	for i in range(2000):
		nn.feedforward()
		nn.backprop()

	outfile = open("HasanAlkhairo_predictions.txt","w")

	for samp in test:
		score = nn.predict(one_hot_encoding(samp))
		outline = samp + "\t" + str(*score) + "\n"
		outfile.write(outline)

	outfile.close()




