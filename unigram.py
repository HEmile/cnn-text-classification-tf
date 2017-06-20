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


def sample_unigram(num_sample):#, X):
  # sample unigram
  unigram_terms, unigram_prob, _ = get_ngram(1, X)
  unigram_sample = np.random.choice(unigram_terms, num_sample, p=unigram_prob)
  
  return unigram_sample

