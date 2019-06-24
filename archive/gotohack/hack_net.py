from IPython import get_ipython
import numpy as np
import theano
import theano.tensor as T
import lasagne
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pyaspeller import Word
get_ipython().magic('env THEANO_FLAGS="device=gpu1"')

train = False
names = pd.read_csv("/mnt/mainstore/data/izvestya.csv").title.values

# should start with magic symbol
start_token = " "
names = [start_token + name for name in names]

token_set = set()
for name in names:
    for letter in name:
        token_set.add(letter)
tokens = list(token_set)

# choosing crop value via plot
# plt.hist(list(map(len, names)), bins=25)

MAX_LEN = min([80, max(list(map(len, names)))])

token_to_id = {t: i for i, t in enumerate(tokens)}
id_to_token = {i: t for i, t in enumerate(tokens)}


if not train:
    with open("last_w.pcl", 'rb') as f:
        trained_weights, tokens, token_to_id = pickle.load(f)

names_ix = list(map(lambda name: list(map(token_to_id.get, name)), names))

for i in range(len(names_ix)):
    names_ix[i] = names_ix[i][:MAX_LEN]  # crop too long
    if len(names_ix[i]) < MAX_LEN:
        names_ix[i] += [token_to_id[" "]] * (MAX_LEN - len(names_ix[i]))  # pad too short

assert len(set(map(len, names_ix))) == 1

names_ix = np.array(names_ix)

# net in lasagne
input_sequence = T.matrix('token sequencea', 'int32')
target_values = T.matrix('actual next token', 'int32')

l_in = lasagne.layers.InputLayer(shape=(None, None), input_var=input_sequence)

l_emb = lasagne.layers.EmbeddingLayer(l_in, len(tokens), 40)

l_rnn1 = lasagne.layers.GRULayer(l_emb, 1024, grad_clipping=5)

l_rnn2 = lasagne.layers.LSTMLayer(l_rnn1, 1024, grad_clipping=5)

l_rnn_flat = lasagne.layers.reshape(l_rnn2, (-1, l_rnn2.output_shape[-1]))

l_out = lasagne.layers.DenseLayer(l_rnn_flat, len(tokens), nonlinearity=lasagne.nonlinearities.softmax)

weights = lasagne.layers.get_all_params(l_out, trainable=True)

network_output = lasagne.layers.get_output(l_out)

predicted_probabilities_flat = network_output
correct_answers_flat = target_values.ravel()

loss = T.mean(lasagne.objectives.categorical_crossentropy(predicted_probabilities_flat, correct_answers_flat))

updates = lasagne.updates.adam(loss, weights)

# training
train = theano.function([input_sequence, target_values], loss, updates=updates, allow_input_downcast=True)

# computing loss without training
compute_cost = theano.function([input_sequence, target_values], loss, allow_input_downcast=True)

# reshape back into original shape
next_word_probas = network_output.reshape((input_sequence.shape[0], input_sequence.shape[1], len(tokens)))
# predictions for next tokens (after sequence end)
last_word_probas = next_word_probas[:, -1]
probs = theano.function([input_sequence], last_word_probas, allow_input_downcast=True)


def correct_phrase(text):
    words = []
    for word in text.split():
        try:
            corrected = Word(word).spellsafe
        except Exception:
            pass
        corrected = corrected if corrected else word
        words.append(corrected)

    return ' '.join(words)


def generate_sample(seed_phrase=None, N=MAX_LEN, t=1, n_snippets=1):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
        
    parameters:
        sample_fun - max_ or proportional_sample_fun or whatever else you implemented
        
        The phrase is set using the variable seed_phrase

        The optional input "N" is used to set the number of characters of text to predict.     
    '''
    if seed_phrase is None:
        seed_phrase = start_token
    if len(seed_phrase) > MAX_LEN:
        seed_phrase = seed_phrase[-MAX_LEN:]
    assert type(seed_phrase) is str

    snippets = []
    for _ in range(n_snippets):
        sample_ix = []
        x = list(map(lambda c: token_to_id.get(c, 0), seed_phrase))
        x = np.array([x])

        for i in range(N):
            # Pick the character that got assigned the highest probability
            p = probs(x).ravel()
            p = p ** t / np.sum(p ** t)
            ix = np.random.choice(np.arange(len(tokens)), p=p)
            sample_ix.append(ix)

            x = np.hstack((x[-MAX_LEN + 1:], [[ix]]))

        random_snippet = seed_phrase + ''.join(id_to_token[ix] for ix in sample_ix)
        snippets.append(random_snippet)

    print(correct_phrase("----\n %s \n----" % '; '.join(snippets)), '/n')


#     return correct_phrase("----\n %s \n----" % '; '.join(snippets))

generate_sample(" abc")


def sample_batch(data, batch_size):
    rows = data[np.random.randint(0, len(data), size=batch_size)]

    return rows[:, :-1], rows[:, 1:]


if train:
    print("Training ...")

    n_epochs = 100
    batches_per_epoch = 500
    batch_size = 10

    for epoch in range(n_epochs):

        print("Generated names")
        generate_sample(n_snippets=10, t=2)

        avg_cost = 0

        for _ in range(batches_per_epoch):
            x, y = sample_batch(names_ix, batch_size)
            avg_cost += train(x, y)

        print("Epoch {} average loss = {}".format(epoch, avg_cost / batches_per_epoch))
        if epoch % 5 == 0:
            with open("last_w_{}.pcl".format(epoch), 'wb') as f:
                pickle.dump([lasagne.layers.get_all_param_values(l_out), tokens, token_to_id], f)
        generate_sample(seed_phrase=" Stepic", n_snippets=10, t=5)
else:
    lasagne.layers.set_all_param_values(l_out, trained_weights)
    generate_sample(seed_phrase=" Предприниматель", n_snippets=10, t=2)
