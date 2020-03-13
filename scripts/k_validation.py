import numpy as np
import math


def k_cross_validation(pos, neg, k):

	if len(pos) != len(neg):
		return "Please used balanced classes"

	if len(pos) % k == 0:
		size = len(pos) / k
	else:
		size = math.ceil(len(pos)/k)

	for i in range(0,len(pos),size):
		tempx = pos.copy()
		tempy = neg.copy()

		zipped = list(zip(tempx,tempy))
		test_on = zipped[i:i+size]

		del zipped[i:i+size]

		unzipped = list(zip(*zipped))
		train_onx, train_ony = unzipped[0],unzipped[1]

		yield train_onx, train_ony, test_on


def test_nn(nerualnet_object, test_set):
	pass