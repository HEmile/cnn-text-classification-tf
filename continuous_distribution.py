import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import data_helpers as dh
from unigram import sample_unigram

class nGrams:
    def __init__(self, X=""):
        if X == "":
            X, y = dh.load_data_and_labels("data/rt-polaritydata/rt-polarity.pos",
                                           "data/rt-polaritydata/rt-polarity.neg")
        self.unigram_terms, self.unigram_prob, _ = get_ngram(1, X)
        self.bigram_terms, self.bigram_prob, _ = get_ngram(2, X)
        self.bigram_dict = {}
        for i in range(len(self.bigram_terms)):
            terms = self.bigram_terms[i].split()
            if terms[0] in self.bigram_dict:
                self.bigram_dict[terms[0]][0].append(terms[1])
                self.bigram_dict[terms[0]][1].append(self.bigram_prob[i])
            else:
                self.bigram_dict[terms[0]] = ([terms[1]], [self.bigram_prob[i]])
        # self.bigram_dict = get_ngramDict
        self.trigram_terms, self.trigram_prob, _ = get_ngram(3, X)


def get_ngram(n, data):
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=0, token_pattern=r"\b\w+\b")  # retain uninformative words
    X = vectorizer.fit_transform(data)
    matrix_terms = np.array(vectorizer.get_feature_names())
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    matrix_prob = matrix_freq / float(np.sum(matrix_freq))
    return matrix_terms, matrix_prob, matrix_freq


# works with either one or two words (resulting in bigrams or trigrams)
def sample_continous(words, ngrams):
    # X, y = dh.load_data_and_labels("data/rt-polaritydata/rt-polarity.pos",
    # "data/rt-polaritydata/rt-polarity.neg")

    # sample trigram
    if len(words) >= 3:
        trigram_terms = ngrams.trigram_terms
        trigram_prob = ngrams.trigram_prob

    # print("words:", words)
    t_index = []
    # resample if no trigram exists given a bigram
    while len(t_index) == 0:
        # sampling bigram
        if len(words) == 2:
            if not words[0] in ngrams.bigram_dict:
                return [words[0], sample_unigram(1)[0]]
            bigrams, bigram_prob = ngrams.bigram_dict[words[0]]
            if words[1] in bigrams:
                windex = bigrams.index(words[1])
                bigrams = list(bigrams)
                del bigrams[windex]
                bigram_prob = list(bigram_prob)
                del bigram_prob[windex]

            if len(bigrams) < 1:
                return [words[0], sample_unigram(1)[0]]
            else:
                bg_prob = bigram_prob / np.sum(bigram_prob)
                bg_sample = np.random.choice(bigrams, 1, p=bg_prob)[0]
                return [words[0], bg_sample]  # uncomment this if a trigram should be based on only one word
        else:
            bigram_sample = words

        if len(bigram_sample) > 0:
            t_index = [j for j in range(len(trigram_terms)) if
                       bigram_sample[0] == trigram_terms[j].split()[0] and
                       bigram_sample[1] == trigram_terms[j].split()[1]]
        else:
            print("WARNING, no trigrams found because no bigrams were found!")
            break

        n_indices = len(t_index)
        if n_indices > 0:
            tg_sample = words
            while n_indices > 0 and (tg_sample[2] == words[2]):
                tg_prob = trigram_prob[t_index] / np.sum(trigram_prob[t_index])
                tg_terms = trigram_terms[t_index]
                tg_sample = np.random.choice(tg_terms, 1, p=tg_prob)
                tg_sample = tg_sample[0].split()
                print("loop")
                n_indices = n_indices - 1
        else:
            # print("WARNING, no trigrams found because no bigrams were found!")
            return ("XXX")
    return tg_sample
