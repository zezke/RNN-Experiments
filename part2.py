import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import random

# Own stuff
from rnnnumpy import RNNNumpy

vocabulary_size = 8000
unknown_token = "UNK"
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"

# Read the data and append the start and end tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-12-15000.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append the start and end tokens
    sentences = ["%s %s %s" % (start_token, x, end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and occurred %d times." % (vocab[-1][0], vocab[-1][1])

# Replace the words not in our vocabulary with the unknown token 
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

random_index = random.randrange(len(sentences))
print "Example sentence: '%s'" % sentences[random_index]
print "Example sentence after preprocessing: '%s'" % tokenized_sentences[random_index]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print an training data example
x_example, y_example = X_train[random_index], y_train[random_index]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculateLoss(X_train[:1000], y_train[:1000])