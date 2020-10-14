import numpy as np

text = "I like second edition and first edition."

# * preprocess : text받아서 corpus [], id_to_word {}, word_to_id {} 출력


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    #! 다시해보기
    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = [word_to_id[word] for word in words]

    return corpus, word_to_id, id_to_word


corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)

# * Create_co_matrix
# * 1 내가 짜본방법


def create_co_matrix(corpus, window_size=1):
    num_words = np.max(corpus)
    matrix = np.zeros((num_words+1, num_words+1), dtype=np.int32)
    print(matrix)
    print(len(corpus))
    for i in range(num_words+1):  # *0~6
        for j in range(len(corpus)):
            if corpus[j] == i:
                if (i != 0):
                    matrix[i, corpus[j-1]] += 1
                if i != num_words:
                    matrix[i, corpus[j+1]] += 1
    return matrix
# *2 책방법


def create_co_matrix1(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx != 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx != corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


matrix, matrix1 = create_co_matrix(corpus), create_co_matrix1(corpus, 7)
print(matrix)
# * 벡터간 유사도


def cos_similarity(x, y, eps=1e-8):
    nx = x/(np.sqrt(np.sum(x**2)) + eps)
    ny = y/(np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


# * edition 과 first사이의 유사도를 확인
c3, c5 = matrix[2], matrix[5]
print(cos_similarity(c3, c5))

# * 가장 유사도 높은 top개수만큼의 단어와 유사도를 출력


def most_similar(query, words_matrix, word_to_id, id_to_word, num=4):
    if query not in word_to_id:
        print("다시!")
        return

    similarity_list = np.zeros(len(id_to_word))
    query_id = word_to_id[query]
    query_vec = words_matrix[query_id]

    for i in range(len(id_to_word)):
        target_vec = words_matrix[i]
        similarity = cos_similarity(query_vec, target_vec)
        similarity_list[i] = similarity
    #!
    count = 0
    for j in (-1*similarity_list).argsort():  # *오름차순 정렬, index를 return
        print("%s: %s" % (id_to_word[j], similarity_list[j]))
        count += 1
        if count > num-1:
            return


# * 확인해보기
most_similar('edition', matrix, word_to_id, id_to_word)
