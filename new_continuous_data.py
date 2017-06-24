from numpy import floor, ceil
from continuous_distribution import nGrams, sample_continous
# from unigram import sample_unigram
from new_data import words_from_ngram_distribution
from cbow_sample import sample_cbow
import pickle

#dummy placefiller function
def word_from_distribution(ngrams, words):
    if len(words) == 1: #"unigram":
        return ngrams.sample_unigram()
    if len(words) == 2: # "bigram":
        return sample_continous(words, ngrams)
    if len(words) == 3: # "trigram":
        return sample_continous(words, ngrams)
    else:
        return(["PLANETS", "PLANETS", "PLANETS"])

def get_mutations(sentence, S, window_size=2, window_mode="preceding", cbow_most_prob=False):
    
    sentence = sentence.split()
    
    if window_mode=="preceding":
        ngrams = nGrams()

    if window_mode=="cbow":
        with open('cbow_model/model.dump', 'rb') as f:
            model = pickle.load(f) #TODO!!!

    mutations = []
    for index in range(len(sentence)+1):
        if window_mode == "preceding" and (index - window_size >= 0):
            for _ in range(S):
                mutation = sentence[:]
                indices = range(index - window_size, index)
                words = mutation[index - window_size:index]
                mutated_words = word_from_distribution(ngrams, words)

                for mutated_index, mutated_word in zip(indices, mutated_words):
                    mutation[mutated_index] = mutated_word

                mutations.append(' '.join(mutation))
        elif window_mode == "preceding" and (index - window_size >= -(window_size-1)):
        #only the words which can not be predicted by ngram (example: bigram -> index 0 or trigram -> index 0, 1)
            mutated_words = words_from_ngram_distribution("unigram", S)

            all_mutated_words = list(mutated_words)
            for _ in range(S):
                mutation = sentence[:]
                mutated_words = all_mutated_words.pop()
                for mutated_index, mutated_word in zip(range(index-1,index), mutated_words.split()):
                    mutation[mutated_index] = mutated_word
                mutations.append(' '.join(mutation))
        elif window_mode == "cbow" and (index - 1 >= 0) and (index < len(sentence)-1): #always uses a window of 3
            mutated_words = sample_cbow(sentence[index-1], sentence[index+1], model, sentence[index], samples=S, most_prob=cbow_most_prob)
            for word in mutated_words:
                mutation = sentence[:]
                mutation[index] = word[0]

                mutations.append(' '.join(mutation))
        elif window_mode == "cbow" and (index == 0):
            mutated_words = sample_cbow("<START>", sentence[index+1], model, sentence[index], samples=S, most_prob=cbow_most_prob)
            for mutated_word in mutated_words:
                mutation = sentence[:]
                mutation[index] = mutated_word[0]

                mutations.append(' '.join(mutation))
        elif window_mode == "cbow" and (index == len(sentence)-1):
            mutated_words = sample_cbow(sentence[index-1], "<END>", model, sentence[index], samples=S, most_prob=cbow_most_prob)
            for mutated_word in mutated_words:
                mutation = sentence[:]
                mutation[index] = mutated_word[0]

                mutations.append(' '.join(mutation))

    return mutations


def get_removed_mutations(sentence):
    mutations = []
    words = sentence.split()
    for i in range(len(words)):
        neww = list(words)
        del neww[i]
        # neww[i] = '-UNK'
        mutations.append(' '.join(neww))
    return mutations

def get_unked_mutations(sentence):
    mutations = []
    words = sentence.split()
    for i in range(len(words)):
        neww = list(words)
        neww[i] = '-UNK'
        mutations.append(' '.join(neww))
    return mutations


def get_removed_phrase_mutations(sentence, max_phrase=3):
    mutations = []
    words = sentence.split()
    for phr_length in range(1, max_phrase + 1):
        unks = ['UNK'] * phr_length
        for i in range(len(words) + 1 - phr_length):
            neww = list(words)
            neww[i:i+phr_length] = unks
            mutations.append(' '.join(neww))
    return mutations


example = False
if example:
    sentence = "a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass . "
    S = 20
    window_size = 2
    mutations = get_mutations(sentence, S, window_size, window_mode="cbow")

    print(sentence+'\n')

    i = S
    for mutation in mutations:  
        i = i-1
        print(mutation)
        if i == 0:
            i = S
            print()