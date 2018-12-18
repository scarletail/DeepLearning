import numpy as np


def vectorize_sequences(sequences, dimentions=10000):
    results = np.zeros(len(sequences), dimentions)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
