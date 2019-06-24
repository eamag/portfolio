# Solution for test task from Yandex https://yadi.sk/d/VginNdcdxUzH7
# Spend 8 hours or so, all the time with numpy data types, trying to increase accuracy
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error
from copy import copy

np.random.seed(2016)
random.seed(2016)


def prob_round(x, rand):
    sign = np.sign(x)
    x = abs(x)
    if rand < x - int(x):
        return sign*int(x+1)
    else:
        return sign*int(x)


def encode_z(z, min, max, rand):
    return prob_round((z - min) / (max - min) * (2 ** 8 - 1), rand)


def encode(arr):
    min, max = arr.min(), arr.max()
    rand = random.random()
    for x in np.nditer(arr, op_flags=['readwrite']):
        x[...] = encode_z(x, min, max, rand)
    return arr  # .astype('uint8')  # numpy int can be slower than float, see https://goo.gl/PqjjzN


def decode_z(z, min, max):
    return z / (2 ** 8 - 1)**2  # * (max - min) + min  # unsolved problem with accuracy, insight is needed


def decode(arr, min, max):
    return np.vectorize(decode_z)(arr, min, max)


def transform(X, W):

    max_row, min_row = np.argmax(W.sum(axis=1)), np.argmin(W.sum(axis=1))
    max_col, min_col = np.argmax(X.sum(axis=0)), np.argmin(X.sum(axis=0))
    min = W[min_row, :] @ X[:, min_col]
    max = W[max_row, :] @ X[:, max_col]

    return decode(np.dot(encode(X), encode(W)), min, max)

if __name__ == '__main__':
    X = np.random.rand(5, 6).astype('float32')
    W = np.random.rand(6, 5).astype('float32')

    true, decoded = np.dot(W, X), transform(copy(W), copy(X))
    print(math.sqrt(mean_squared_error(true, decoded)))
