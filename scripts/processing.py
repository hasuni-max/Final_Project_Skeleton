import numpy as np
import random 

def one_hot_encoding(dna_string):

	one_hot = np.zeros((len(dna_string),5))

	for index, i in enumerate(dna_string):
		if i == "A":
			one_hot[index,0] = 1
		elif i == "T":
			one_hot[index,1] = 1
		elif i == "G":
			one_hot[index,2] = 1
		elif i == "C":
			one_hot[index,3] = 1
		else:
			one_hot[index,4] = 1
	return one_hot.flatten()

def decoder(input_array):
	max_index = np.argmax(input_array, axis=1)
	decode_dict = {0:"A",1:"T",2:"G",3:"C",4:"N"}

	out_string = [decode_dict[i] for i in max_index]
	return "".join(out_string)

def hamming_distance(str1,str2):

	distance = sum([ 1 if a != b else 0 for a,b in zip(str1,str2)])

	return distance


def parse_fasta(filename):
	sequences = {}

	with open(filename) as fh:
		seq = ""
		for line in fh:
			if line.startswith(">"):
				
				if len(seq) > 0:
					sequences[header] = seq
					seq = ""

				header = line.strip()

			else:
				seq += line.strip().upper()

	return sequences


def generate_negative_samples(filename,kmer_length,number_of_samples,positive_list):
	
	seqs = parse_fasta(filename)

	x = 0
	while x < number_of_samples:
		a_key = random.choice(list(seqs))
		temp_seq = seqs[a_key]
		start = random.randrange(0,len(temp_seq)-kmer_length)
		out_string = temp_seq[start:start+kmer_length]
		if out_string in positive_list:
			continue
		x += 1
		yield temp_seq[start:start+kmer_length]





if __name__ == "__main__":
	
	# print(one_hot_encoding("ATCCN"))

	print(parse_fasta("../data/yeast-upstream-1k-negative.fa"))
