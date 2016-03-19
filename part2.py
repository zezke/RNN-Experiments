import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import random
import timeit

# Own stuff
from rnnnumpy import RNNNumpy
from rnntheano import RNNTheano
from utils import *

vocabulary_size = 8000
unknown_token = "UNK"
start_token = "SENTENCE_START"
end_token = "SENTENCE_END"

def trainWithSgd(model, X_train, y_train, learningRate=0.005, nepoch=1, evaluateLossAfter=5):
    # We keep track of the losses so we can plot them later
    losses = []
    numExamplesSeen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluateLossAfter == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((numExamplesSeen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after numExamplesSeen=%d epoch=%d: %f" % (time, numExamplesSeen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learningRate = learningRate * 0.5  
                print "Setting learning rate to %f" % learningRate
            sys.stdout.flush()
            # ADDED! Saving model parameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hiddenDim, model.wordDim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learningRate)
            numExamplesSeen += 1

# Sample some sentences
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

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
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print an training data example
x_example, y_example = X_train[random_index], Y_train[random_index]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)

# Train the RNN
np.random.seed(10)
model = RNNTheano(vocabulary_size, hiddenDim=50)
losses = trainWithSgd(model, X_train, Y_train, nepoch=10, evaluateLossAfter=1)
 
num_sentences = 10
senten_min_length = 7
 
for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)