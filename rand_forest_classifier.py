import data_getter
from sklearn.ensemble import RandomForestClassifier
def main():
	
	data_loc = 'no_name_counts_actresses.txt'
	samples_per_career = 5
	min_length = 10
	min_am = 5
	training_number = 10000
	
	sample_careers, sample_labels = data_getter.get_sample_paths(data_loc, samples_per_career, min_length, min_am)
	
	training_careers = sample_careers[0:training_number]
	training_labels = sample_labels[0:training_number]
	testing_careers = sample_careers[training_number:]
	testing_labels = sample_labels[training_number:]
	
	#Null model first.
	null_score = null_model_classifier(testing_careers, testing_labels)
	print('null score: ', null_score)
	
	#Now ML model.
	model = RandomForestClassifier(class_weight='balanced')
	model.fit(training_careers, training_labels)
	
	score = model.score(testing_careers, testing_labels)	
	print('model score: ', score)


def null_model_classifier(series, labels):
	
	score = 0.0
	for x in range(len(series)):
		am_index = data_getter.get_am(list(series[x]), True)
		if am_index == len(series[x])-1:
			if labels[x] == 1:
				score += 1.0
		else:
			if labels[x] == 0:
				score += 1.0
	score /= len(series)
	
	return score
			

if __name__ == "__main__":
	main()