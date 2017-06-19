import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import data_helpers as dh
X, y = dh.load_data_and_labels("data/rt-polaritydata/rt-polarity.pos",
                                    "data/rt-polaritydata/rt-polarity.neg")
def get_ngram(n, data):
    vectorizer = CountVectorizer(ngram_range=(n,n), min_df=0, token_pattern=r"\b\w+\b") # retain uninformative words
    X = vectorizer.fit_transform(data)
    matrix_terms = np.array(vectorizer.get_feature_names())
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    matrix_prob = matrix_freq / float(np.sum(matrix_freq))
    return matrix_terms, matrix_prob, matrix_freq


def sample_ngram(num_sample, X):
  # sample unigram
  unigram_terms, unigram_prob, _ = get_ngram(1, X)
  unigram_sample = np.random.choice(unigram_terms, num_sample, p=unigram_prob)
  # sample bigram
  bigram_terms, bigram_prob, _ = get_ngram(2, X)
  bigram_sample = []
  # sample trigram
  trigram_terms, trigram_prob, _ = get_ngram(3, X)
  trigram_sample = []
  for i in range(num_sample):
    # sampling bigram
    b_index = [j for j in range(len(bigram_terms)) if unigram_sample[i] == bigram_terms[j].split()[0]]
    bg_prob = bigram_prob[b_index] / np.sum(bigram_prob[b_index])
    bg_terms = bigram_terms[b_index]
    bg_sample = np.random.choice(bg_terms, 1, p=bg_prob)
    bigram_sample.append(bg_sample[0])

    t_index = []
    # resample if no trigram exists given a bigram
    while len(t_index) == 0:
      t_index = [j for j in range(len(trigram_terms)) if
                 bigram_sample[i].split()[0] == trigram_terms[j].split()[0] and
                 bigram_sample[i].split()[1] == trigram_terms[j].split()[1]]
      b_index = [j for j in range(len(bigram_terms)) if unigram_sample[i] == bigram_terms[j].split()[0]]
      bg_prob = bigram_prob[b_index] / np.sum(bigram_prob[b_index])
      bg_terms = bigram_terms[b_index]
      bg_sample = np.random.choice(bg_terms, 1, p=bg_prob)
      bigram_sample[i] = bg_sample[0]
    tg_prob = trigram_prob[t_index] / np.sum(trigram_prob[t_index])
    tg_terms = trigram_terms[t_index]
    tg_sample = np.random.choice(tg_terms, 1, p=tg_prob)
    trigram_sample.append(tg_sample[0])
  return unigram_sample, np.array(bigram_sample), np.array(trigram_sample)