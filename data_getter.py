import random
import numpy as np

#Program gets actor/actress careers and attempts pre/post AM classification in general framework.
#Uses padding to ensure sequences are all the same length.

def main():
	#Testing main, wont be called externally.
	data_loc = '/Users/oliverwilliams/Documents/qmul/data/imdb/no_name_counts_actors.txt'
	samples_per_career = 5
	min_length = 5
	min_am = 3
	
	sample_careers, sample_labels = get_sample_paths(data_loc, samples_per_career, min_length, min_am)	
	
	return


def get_sample_paths(data_loc, samples_per_career, min_career_length, min_am):
	
	f = open(data_loc,'r')
	
	sample_series = []
	label_series = []
	
	for line in f:
		if len(line) > 1:
			int_series = [int(x) for x in line.split('\t')[0:-1]]
			if sum(1 for w in int_series if w!=0) > min_career_length and max(int_series) > min_am:
				#Now get sub_samples from the career.
				sub_samples, sub_label_series = get_sub_samples(int_series, samples_per_career)
				sample_series += sub_samples
				label_series += sub_label_series
	
	#Got paths, now pad and process them.
	sample_series, label_series = pad_and_process(sample_series, label_series)
	
	#print(sample_series.shape, label_series.shape)
	
	return sample_series, label_series


#Last value determines if we take the final AM (True) or the first (False).
def get_sub_samples(series, samples_per_career, last_value=True):
	
	am_value = get_am(series, last_value) #Get the AM for a sequence.
	
	samples = []
	labels = []
	for i in range(samples_per_career):
		loc = random.randint(1,len(series))			#Pick a random location to sample up to.
		samp_seq = series[0:loc]
		lable = 1 if loc <= am_value else 0		#Get class lable - 1 for pre AM, 0 for post AM.
		samples.append(samp_seq)
		labels.append(lable)
	
	return samples,labels


def get_am(series, last_value):
	
	if last_value:	
		rev = series[::-1]		#Reverse list.
		am_rev = rev.index(max(series))			#Get the location of the last AM (first in reversed sequence).
		am_val = len(series) - am_rev - 1			#Index of last AM in foward sequence.
		return am_val
	else:
		am_val = X.index(max(series))
		return am_val
	

def pad_and_process(series_set, label_set):
	
	new_comb_series = []
	max_len = max([len(x) for x in series_set])
	for i in range(len(series_set)):
		zeros_added = max_len - len(series_set[i])
		new_entry = ([0]*zeros_added) + series_set[i] + [label_set[i]] 
		new_comb_series.append(new_entry)
	
	new_comb_series = np.asarray(new_comb_series)
	np.random.shuffle(new_comb_series)
	
	new_series = new_comb_series[:,:max_len]
	label_series = new_comb_series[:,max_len:].reshape((-1,))
	
	return new_series, label_series
		
	
		

if __name__ == "__main__":
	main()