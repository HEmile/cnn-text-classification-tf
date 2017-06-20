from numpy import floor, ceil
from continuous_distribution import nGrams, sample_continous
from unigram import sample_unigram
from new_data import words_from_ngram_distribution

#dummy placefiller function
def word_from_distribution(ngrams, words):
    # ug, bg, tg = sample_ngram(1)
    # if distribution == "unigram":
    if len(words) == 1:
        return unigram
    # if distribution == "bigram":
    if len(words) == 2:
        return sample_continous(words, ngrams)
    # if distribution == "trigram":
    if len(words) == 3:
        return sample_continous(words, ngrams)
    else:
        return(["PLANETS", "PLANETS", "PLANETS"])

#only works with uneven window_size (as in, symmetrical around one index)
def get_mutations(sentence, S, window_size=2, window_mode="preceding"):
    ngrams = nGrams()
    sentence = sentence.split()

    # print("mutating sentence: ", sentence, '\n')

    mutations = []
    for index in range(len(sentence)+1):
        # mutation = sentence[:]
        indices = []
        words = []
        mutated_words = []
        # if window_mode == "symmetric" and ((index - int(floor(window_size/2))) >= 0) and ((index + int(ceil(window_size/2))) <= len(sentence)): #only symmetric
        #     indices = range(index - int(floor(window_size/2)), (index + int(ceil(window_size/2))))
        #     words = mutation[(index - int(floor(window_size/2))):(index + int(ceil(window_size/2)))]
        #     mutated_words = word_from_distribution(ngrams, words) #SKIPGRAM (not really, but what we call a skipgram)
        # print("index:", index)
        # print("index-window_size:", index-window_size)
        if window_mode == "preceding" and (index - window_size >= 0):
            for _ in range(S):
                mutation = sentence[:]
                indices = range(index - window_size, index)
                words = mutation[index - window_size:index]
                mutated_words = word_from_distribution(ngrams, words)
            # if window_mode == "following" and (index + window_size <= len(sentence)):
            #     indices = range(index, index + window_size)
            #     words = mutation[index:index + window_size]
            #     mutated_words = word_from_distribution(ngrams, words) #weet nog niet hoe!

            # print("original words", words)
            # print("mutated  words", mutated_words)

                for mutated_index, mutated_word in zip(indices, mutated_words):
                    mutation[mutated_index] = mutated_word

            # print(indices)
                # print(' '.join(mutation))
                mutations.append(' '.join(mutation))
        elif window_mode == "preceding" and (index - window_size == -1):
            #first word unigram for bigram
            mutated_words = words_from_ngram_distribution("unigram", S)

            all_mutated_words = list(mutated_words)
            for _ in range(S):
                mutation = sentence[:]

                mutated_words = all_mutated_words.pop()
                for mutated_index, mutated_word in zip(range(0,1), mutated_words.split()):
                    mutation[mutated_index] = mutated_word

                mutations.append(' '.join(mutation) + "***")

    return mutations


example = False
if example:
    sentence = "a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass . "
    S = 2
    window_size = 2
    mutations = get_mutations(sentence, S, window_size)

    print(sentence+'\n')

    i = S
    for mutation in mutations:
        i = i-1
        print(mutation)
        if i == 0:
            i = S
            print()

