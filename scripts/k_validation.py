import numpy as np
import math


def k_cross_validation(pos, neg, k):


	if len(pos) != len(neg): #hacky, but whatever for now
		return "Please used balanced classes"

	if len(pos) % k == 0: #determines window size
		size = len(pos) / k
	else:
		size = math.ceil(len(pos)/k)

	for i in range(0,len(pos),size): #grab windows of size, size. 
		tempx = pos.copy() #we want to make sure to copy these lists and not just create new pointers
		tempy = neg.copy()

		zipped = list(zip(tempx,tempy)) #Zip pos and neg. I do this to ensure that our classes stay balanced
		test_on = zipped[i:i+size] #save this window as the test set

		del zipped[i:i+size] # and remove the test set from the zipped list

		unzipped = list(zip(*zipped)) #This is the training set now 
		train_onx, train_ony = unzipped[0],unzipped[1] #seperate out negative and positive 

		yield train_onx, train_ony, test_on #yield a positive list, negative list and the testing for this fold
