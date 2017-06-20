from numpy import floor, ceil
from ngram import sample_ngram
from unigram import sample_unigram

def words_from_ngram_distribution(distribution, sample_count=1):
    if distribution == "unigram" or distribution == "bigram" or distribution == "trigram":
        if distribution == "unigram":
            ug = sample_unigram(sample_count)
            return ug
        if distribution == "bigram":
            return bg
        if distribution == "trigram":
            return tg
    else:
        print("The ngram implementation uses either unigram, bigram or trigram")
        return(["PLANETS", "PLANETS", "PLANETS"])

def count_samples(sentence, S, window_size):
    sample_count = 0
    for index in range(len(sentence)):
        if ((index - int(floor(window_size/2))) >= 0) and ((index + int(ceil(window_size/2))) <= len(sentence)):
            sample_count += 1
    return sample_count * S

def get_S_ngram_mutations(sentence, S, distribution):
    if distribution == "unigram": window_size = 1
    if distribution == "bigram": window_size = 2
    if distribution == "trigram": window_size = 3


    sentence = sentence.split()

    n_samples = count_samples(sentence, S, window_size)
    mutated_words = words_from_ngram_distribution(distribution, n_samples)

    all_mutated_words = list(mutated_words)

    mutations = []
    for index, word in enumerate(sentence):
        for _ in range(S):
            if ((index - int(floor(window_size/2))) >= 0) and ((index + int(ceil(window_size/2))) <= len(sentence)):
                mutation = sentence[:]
                indices = range(index - int(floor(window_size/2)), (index + int(ceil(window_size/2))))

                mutated_words = all_mutated_words.pop()
                for mutated_index, mutated_word in zip(indices, mutated_words.split()):
                    mutation[mutated_index] = mutated_word

                mutations.append(' '.join(mutation))
    return mutations

example = False
#example of how to use the function
if example:
    sentence = "a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass . "
    distribution = "unigram"
    mutations = get_S_ngram_mutations(sentence, 3, distribution)
    print("mutating:", sentence, '\n')
    for mutation in mutations:
        print(mutation)
