import sys
from collections import Counter, defaultdict
import re
import math
from stemming.porter2 import stem
import nltk
from itertools import islice, izip, tee
from sklearn.metrics import confusion_matrix
from nltk.metrics import BigramAssocMeasures

""" TOKENIZATION """

def tokenize_porter(title, body):
	""" Break text into words and stem user porter stemmer """
	# break up words & remove stopwords
	title_break = stopWords(nltk.word_tokenize(title), lower_case = True)
	body_break = stopWords(nltk.word_tokenize(body), lower_case = True)
	#print title_break
	return(["title:" + stem(title) for title in title_break] + ["body:" + stem(body) for body in body_break])

""" TEXT PRE-PROCESSING """

def stopWords(text, lower_case = False):
	""" Remove stop words """
	# If lower_case == true
	if(lower_case == True):
		text = [word.lower() for word in text]
	# Remove stopwords
	return([word for word in text if word not in nltk.corpus.stopwords.words('english') + [unicode("youre"), unicode("may")]])

def numbers(text):
	""" remove numbers and percentages """
	return(re.sub(r'[0-9%]','', text))

def clean(line):
	""" Remove whitespace and numbers """
	# Strip whitespace
	text = re.sub(r'[^\w\s]','', line)
	# Strip numbers
	text = numbers(text)
	# strip and join
	return(" ".join(text.split()))

""" FEATURE SELECTION METHODS """

def chiSQ(priors, likelihood, keep):
	""" Extract the 10000 most informative features using chi-square """
	words = {}
	# Total word count
	twc = sum(priors.values())
	# All words in the counter
	words_unique = [likelihood[section].keys() for section in likelihood.keys()]
	words_unique = set(sum(words_unique, []))
	for word in words_unique:
		# Go past each class
		scores = []
		for c in priors.keys():
			# Class word count
			cwc = priors[c]
			# Get word occurrence over all classes
			totalFreq = sum([likelihood[section][word] for section in priors.keys()])
			# Word count within class
			wc = likelihood[c][word]
			# Get chi-sq
			score = BigramAssocMeasures.chi_sq(wc, (totalFreq, cwc), twc)
			# Append
			scores.append(score)
		# Add to dict
		words[word] = sum(scores)
	# Select best words
	bestWords = sorted(words.iteritems(), key=lambda (w,s): s, reverse=True)[:keep]
	# Save
	with open("chiSQ.txt", 'w') as f:
		print >> f, bestWords
	# Get names
	bestWords = [b[0] for b in bestWords]
	# Filter likelihood
	for c in priors.keys():
		for key in list(likelihood[c]):
			if key not in bestWords:
				del likelihood[c][key]
	# Return
	return(likelihood)

""" NAIVE BAYES ALGORITHM """

def priors_and_likelihood(lines, keep, word_filter = "none"):
	""" Calculate priors (== group sizes) and likelihood of each word (== occurrences) """
	# Open counters
	priors = Counter()
	likelihood = defaultdict(Counter)
	# For each headline & snippet, tokenize and add word to likelihood counter
	for line in lines:
		priors[line[1]] += 1
		for word in tokenize_nltk(clean(line[2]), clean(line[3])):
			likelihood[line[1]][word] += 1
	# Filter
	if word_filter == "none":
		return(priors, likelihood)
	if word_filter == "chi-sq":
		likelihood = chiSQ(priors, likelihood, keep)
	# Return
	return(priors, likelihood)

def naive_bayes(tokens, priors, likelihood):
	""" Return the class that maximizes the posterior """
	# Get prediction
	max_class = (-1E6, '')
	for c in priors.keys():
		p = math.log(priors[c])
		n = float(sum(likelihood[c].values()))
		for word in tokens:
			p = p + math.log(max(1E-6, likelihood[c][word]) / n)

		if p > max_class[0]:
			max_class = (p,c)

	return max_class[1]

""" HELPERS """

def read_training_file(filename):
	""" Open tab-delimited file, read each line, split and return """
	with open(filename) as f:
		return([line.split("\t") for line in f])

def tokenize_testset(line):
	# Tokenize
	tokenized_text = tokenize_nltk(clean(line[2]), clean(line[3]))
	# Return
	return(tokenized_text)

def read_testing_file(filename):
	""" Open tab-delimited file, read each line, split and return """
	with open(filename) as f:
		return([line.split("\t") for line in f])

def savePredictions(filename, labels, predictions):
	# Open filename and write
	with open(filename, 'w') as f:
		for f1,f2 in zip(labels, predictions):
			print >> f,f1 + "\t" + f2

""" CROSS-VALIDATION """

def KfoldCV(data, folds):
	""" Split data in training / test for number of folds and estimate test error """
	# Open list for stats
	stats = list()
	# Subset
	subsetL = len(data) / folds
	for i in range(folds):
		# subset
		testing_cv = data[i*subsetL:][:subsetL]
		training_cv = data[:i*subsetL] + data[(i+1)*subsetL:]
		# train
		(priors, likelihood) = priors_and_likelihood(training_cv, 5000, word_filter = "chi-sq")
		# test set labels
		labels = [line[1] for line in testing_cv]
		# Tokenize
		tokens_test = [tokenize_testset(line) for line in testing_cv]
		# predictions
		pred = [naive_bayes(token, priors, likelihood) for token in tokens_test]
		# Calculate confusion matrix
		#stats.append(confusion_matrix(labels, pred))
		accuracy = 0
		for i in range(1,len(labels)):
			if(labels[i] == pred[i]):
				accuracy += 1
			# Add to stats
		stats.append(float(accuracy) / len(labels))
	# Return
	return(stats)

""" TOP-LEVEL """

def main():
	# Arguments
	training_file = sys.argv[1]
	testing_file = sys.argv[2]
	CV = sys.argv[3]
	folds = int(sys.argv[4])
	save_model = sys.argv[5]
	# Read training
	lines_train = read_training_file(training_file)
	# Cross validation
	if(CV == "k-fold") :
		# K-fold CV
		CVres = KfoldCV(lines_train, folds)
		est2 = [str(e) for e in CVres]
		print "Results for K-fold cross-validation with k = 5 folds " + ", ".join(est2) + " with an overall accuracy of " + str(sum(CVres) / len(CVres))
	# Priors and likelihood
	(priors, likelihood) = priors_and_likelihood(lines_train, 15000, word_filter = "chi-sq")
	# Read lines
	lines = read_testing_file(testing_file)
	# Tokenize test
	tokens_test = [tokenize_testset(line) for line in lines]
	# Counter
	# test set labels
	labels = [line[1] for line in lines]
	# predictions
	pred = [naive_bayes(token, priors, likelihood) for token in tokens_test]
	# Save model if specified
	if(save_model != "FALSE" and save_model.endswith(".txt")):
		# Save labels and pred to txt
		savePredictions(save_model, labels, pred)
	# Calculate confusion matrix
	#stats.append(confusion_matrix(labels, pred))
	accuracy = 0
	for i in range(1,len(labels)):
		if(labels[i] == pred[i]):
			accuracy += 1

	print "Calibration/test: Classified %d correctly out of %d for an accuracy of %f"%(accuracy, len(labels), float(accuracy) / len(labels))

if __name__ == "__main__":
main()
