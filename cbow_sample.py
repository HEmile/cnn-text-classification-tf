from gensim import models
import numpy as np
import pickle

PATH = 'data/rt-polaritydata/combined.txt'
MODEL_PATH = 'cbow_model/'
NUM_TOKENS = 268215
TRAIN_MODEL = False
LOAD_MODEL = True


def train_cbow(embedding_size):
    sentences = models.word2vec.LineSentence(PATH)
    model = models.Word2Vec(sentences, size=embedding_size, window=1, min_count=0, workers=4)
    return model


def sample_cbow(left_word, right_word, model, accurate_word, samples=1, most_prob=False):
    output_word = model.predict_output_word([left_word, right_word], topn=NUM_TOKENS)
    cbow_terms = []
    cbow_prob = []
    norm_term = 0  # used to renormalize probability
    cbow_samples = []
    if not most_prob:
        for i in range(len(output_word)):
            norm_term += output_word[i][1]
            word = output_word[i][0]
            if word == accurate_word: continue
            cbow_terms.append(word)
            cbow_prob.append(output_word[i][1])
        cbow_prob = cbow_prob / np.sum(cbow_prob)
        for _ in range(samples):
            cbow_samples.append(np.random.choice(cbow_terms, 1, p=cbow_prob))
    else:
        cbow_samples = [output_word[0][0]] * samples  # return word that has the highest probability predicted by cbow
    return cbow_samples


model = None
if TRAIN_MODEL:
    model = train_cbow(128)
    pickle.dump(model, open('cbow_model/model.dump', 'wb'))
if LOAD_MODEL:
    model = pickle.load(open('cbow_model/model.dump', 'rb'))
# print(sample_cbow('i', 'juanna', model))
