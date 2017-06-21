from numpy import floor, ceil
from continuous_distribution import nGrams, sample_continous
# from unigram import sample_unigram
from new_data import words_from_ngram_distribution

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

def get_mutations(sentence, S, window_size=2, window_mode="preceding"):
    ngrams = nGrams()
    sentence = sentence.split()

    mutations = []
    for index in range(len(sentence)+1):
        indices = []
        words = []
        mutated_words = []
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
    return mutations


example = True
if example:
    sentence = "a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass . "
    S = 20
    window_size = 3
    mutations = get_mutations(sentence, S, window_size)

    print(sentence+'\n')

    i = S
    for mutation in mutations:
        i = i-1
        print(mutation)
        if i == 0:
            i = S
            print()