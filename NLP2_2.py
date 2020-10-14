from common.util import preprocess, create_co_matrix, cos_similarity, ppmi
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')

# * 'a is a apple'
Matrix = np.array([[0, 2, 1], [2, 0, 0], [1, 0, 0]])

# * PPMI지수 구하기

text = 'I learned analysis equation and synthesis equation of Fourier transfrom I is like laplace transform'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)


def ppmi1(C, eps=1e-8):
    N = np.sum(C)/2.0
    M = np.zeros_like(C)
    S = np.sum(C, axis=0)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            term1 = np.log2(C[i, j]*N/(S[i]*S[j])+eps)
            M[i, j] = max(0, term1)
    return M


W = ppmi(C)
# * SVD!
U, S, V = np.linalg.svd(W)
print(U.shape)
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.3)
# plt.show()
