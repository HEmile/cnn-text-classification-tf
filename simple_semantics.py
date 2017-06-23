
def addCounts(label_dict, reviews, label):
    for review in reviews:
        for word in review.split():
            try:
                label_dict[word][0] += label[0]
                label_dict[word][1] += label[1]
            except:
                label_dict[word] = label[:]
    return label_dict

def create_label_dict():
    with open("data/rt-polaritydata/rt-polarity.pos", 'r') as pos:
        positive_reviews = pos.readlines()

    with open("data/rt-polaritydata/rt-polarity.neg", 'r') as neg:
        negative_reviews = neg.readlines()

    label_dict = {}

    addCounts(label_dict, positive_reviews, [1, 0])
    addCounts(label_dict, negative_reviews, [0, 1])

    return label_dict

def try_sentence(label_dict, sentence):
    pos_probs = []
    neg_probs = []
    for word in sentence.lower().split():
        try:
            pos_probs.append(1.0*label_dict[word][0]/(label_dict[word][0]+label_dict[word][1]))
            neg_probs.append(1.0*label_dict[word][1]/(label_dict[word][0]+label_dict[word][1]))
        except: #if unknown word
            pos_probs.append(1.0)
            neg_probs.append(1.0)

    pos_prob = 1.0
    for prob in pos_probs:
        pos_prob = pos_prob * prob

    neg_prob = 1.0
    for prob in neg_probs:
        neg_prob = neg_prob * prob

    if pos_prob > neg_prob:
        return 1, pos_prob, pos_probs
    else:
        return 0, neg_prob, neg_probs

label_dict = create_label_dict()

sentence = "effective but too-tepid biopic ."
sentence = "Return of the Jedi is my favorite of the Original Trilogy ."
sentence = "This is a real lame movie that tries too hard to incorporate too many things at once ."
sentence = "When I saw this movie I was stunned by what a great movie it was ."
classification, sentence_prob, word_probs = try_sentence(label_dict, sentence)

print("sentence classified as:", classification)
print("likelihood --- word")
print("-------------------")
for word_prob, word in zip(word_probs, sentence.split()):
    print('{0:10f} ==> {1}'.format(word_prob, word))


# print(label_dict)

