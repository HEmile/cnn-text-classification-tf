import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import data_helpers as dh
# from unigram import sample_unigram

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
        self.trigram_dict = {}
        for i in range(len(self.trigram_terms)):
            terms = self.trigram_terms[i].split()
            if terms[0] in self.trigram_dict:
                if terms[1] in self.trigram_dict[terms[0]]:
                    self.trigram_dict[terms[0]][terms[1]][0].append(terms[2])
                    self.trigram_dict[terms[0]][terms[1]][1].append(self.trigram_prob[i])
                else:
                    self.trigram_dict[terms[0]][terms[1]] = ([terms[2]], [self.trigram_prob[i]])
            else:
                self.trigram_dict[terms[0]] = {}
                self.trigram_dict[terms[0]][terms[1]] = ([terms[2]], [self.trigram_prob[i]])

    def sample_unigram(self, num_sample=1):
        return np.random.choice(self.unigram_terms, num_sample, p=self.unigram_prob)



def get_ngram(n, data):
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=0, token_pattern=r"\b\w+\b")  # retain uninformative words
    X = vectorizer.fit_transform(data)
    matrix_terms = np.array(vectorizer.get_feature_names())
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    matrix_prob = matrix_freq / float(np.sum(matrix_freq))
    return matrix_terms, matrix_prob, matrix_freq


# works with either two or three words (resulting in bigrams or trigrams)
def sample_continous(words, ngrams):

    # sampling bigram
    if len(words) == 2:
        if not words[0] in ngrams.bigram_dict:
            return [words[0], ngrams.sample_unigram()[0]]
        bigrams, bigram_prob = ngrams.bigram_dict[words[0]]
        if words[1] in bigrams:
            windex = bigrams.index(words[1])
            bigrams = list(bigrams)
            del bigrams[windex]
            bigram_prob = list(bigram_prob)
            del bigram_prob[windex]

        if len(bigrams) < 1:
            return [words[0], ngrams.sample_unigram()[0]] #unigram sampling if no word was predicted
        else:
            bg_prob = bigram_prob / np.sum(bigram_prob)
            bg_sample = np.random.choice(bigrams, 1, p=bg_prob)[0]
            return [words[0], bg_sample]  # uncomment this if a trigram should be based on only one word

    # sampling trigrams
    if len(words) == 3:
        if not words[0] in ngrams.trigram_dict:
            return [words[0], words[1], ngrams.sample_unigram()[0]] #unigram sampling if no word was predicted
        if not words[1] in ngrams.trigram_dict[words[0]]:
            return [words[0], words[1], ngrams.sample_unigram()[0]] #unigram sampling if no word was predicted
        trigrams, trigram_prob = ngrams.trigram_dict[words[0]][words[1]]
        if words[2] in trigrams:
            windex = trigrams.index(words[2])
            trigrams = list(trigrams)
            del trigrams[windex]
            trigram_prob = list(trigram_prob)
            del trigram_prob[windex]

        if len(trigrams) < 1:
            return [words[0], words[1], ngrams.sample_unigram()[0]] #unigram sampling if no word was predicted (other then the original word)
        else:
            tg_prob = trigram_prob / np.sum(trigram_prob)
            tg_sample = np.random.choice(trigrams, 1, p=tg_prob)[0]
            return [words[0], words[1], tg_sample]

    return "OEI"
