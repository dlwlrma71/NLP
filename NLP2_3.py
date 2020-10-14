from sklearn.utils.extmath import randomized_svd
from dataset import ptb
from common.util import preprocess, create_co_matrix, ppmi, most_similar
import matplotlib.pyplot as plt
import numpy as np
import sys
import docx2txt
sys.path.append('..')
text = docx2txt.process('/Users/dlwlrma71/Desktop/영어hw.docx')

corpus, word_to_id, id_to_word = preprocess(text)  # * 엑셀 데이터
vocab_size = len(word_to_id)  # * 전체 size!
C = create_co_matrix(corpus, vocab_size, window_size=2)
W = ppmi(C, verbose=True)  # 이거 하고있는중!
wordvec_size = 100
U, S, V = randomized_svd(W, n_components=wordvec_size,
                         n_iter=5, random_state=None)
# * n_components : 감소시킬 디멘션값! n_iter: 반복수

print(U.shape, S.shape)

# except ImportError:
#     U, S, V = np.linalg.svd(W)

wordvecs = U[:, :wordvec_size]  # * 요건 np.linalg.svd썼을때! 없어도된다 사이킷런쓰면

query_list = ['you', 'year', 'could', 'person']
for query in query_list:  # *가장 similar한 거 표시!
    most_similar(query, word_to_id, id_to_word, wordvecs, top=5)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))  # * 각 지점에 이름 매칭

plt.scatter(U[:, 0], U[:, 1], alpha=0.1)  # *plot
plt.show()
