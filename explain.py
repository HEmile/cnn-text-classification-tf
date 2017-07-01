import tensorflow as tf
import os
from tensorflow.contrib import learn
import data_helpers
import numpy as np
import csv
import new_continuous_data
import math

def softmax(scores):
    s1, s2 = scores[0], scores[1]
    e1 =  math.exp(s1)
    e2 = math.exp(s2)
    return e1/(e1+e2), e2/(e1+e2)

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1497534303/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)


def predict(mutations):
    x_variants = np.array(list(vocab_processor.transform(mutations)))
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            probabilities = graph.get_operation_by_name("output/scores").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helpers.batch_iter(list(x_variants), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = []

            for x_test_batch in batches:
                results = sess.run([predictions, probabilities], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, results[0]])
                for result in results[1]:
                    all_probabilities.append(result)
    return all_probabilities, all_predictions


def we(pcminx, pc):
    log1 = math.log(999999999999) if pc >= 1 else math.log(pc / (1-pc))
    log2 = math.log(999999999999) if pcminx >= 1 else math.log(pcminx / (1-pcminx))
    return log1 - log2


def colour_sentence(stacks, normalise_constant=20):
    s = ''
    for stack in stacks:
        norm = max(-1, min(1, float(stack[1]) / normalise_constant))
        if norm < 0:
            s += "{\\color[rgb]{0, 0, " + str(-norm) + "}" + stack[0] + "} "
        else:
            s += "{\\color[rgb]{" + str(norm) + ", 0, 0}" + stack[0] + "} "
    return s


def explain(sentence, mutation_method):
    words = sentence.split()
    n = len(words)
    x_v = []
    prob_distr = []
    for i in range(n):
        xs, probs = mutation_method(sentence, i)
        x_v.extend(xs)
        if not(probs is None):
            prob_distr.extend(probs)
    S = math.ceil(len(x_v) / n)
    x_v.append(sentence)

    all_probabilities, all_predictions = predict(x_v)
    predicted_y = all_predictions[-1]
    # print(predicted_y)
    prob_y = softmax(all_probabilities[-1])[int(predicted_y)]

    weight_evidence = []

    for i in range(n):
        pred_probs = []
        for j in range(S * i, S * (i + 1)):
            # print(x_v[j])
            # print(all_probabilities[j])
            # print(softmax(all_probabilities[j])[int(predicted_y)])
            evidence = we(softmax(all_probabilities[j])[int(predicted_y)], prob_y)
            if len(prob_distr) > 0:
                pred_probs.append(evidence * prob_distr[j])
            else:
                pred_probs.append(evidence)
        if len(prob_distr) > 0:
            weight_evidence.append(sum(pred_probs))
        else:
            weight_evidence.append(np.average(pred_probs))
    stacked = np.column_stack((words, weight_evidence))
    print(colour_sentence(stacked))
    # print(prob_y)
    # print('Predicted class label:', predicted_y)

def explain_window(sentence, mutation_method, samples=600, window_size=7, use_prob_distr=False):
    split = sentence.split()
    n = len(split)
    windows = n - window_size + 1
    x_v = []
    prob_distr = []
    for l in range(windows):
        window = ' '.join(split[l:l+window_size])
        x_v.append(window)
        if use_prob_distr:
            prob_distr.append(0)
        for j in range(window_size):
            mutated_s, prob_d = mutation_method(window, j)
            x_v.extend(mutated_s)
            if not(prob_d is None):
                prob_distr.extend(prob_d)
    x_v.append(sentence)

    all_probabilities, all_predictions = predict(x_v)
    predicted_y = all_predictions[-1]
    prob_y = softmax(all_probabilities[-1])[int(predicted_y)]

    pred_probs = [[] for _ in range(n)]
    c = 0
    for l in range(windows):
        window_prob = softmax(all_probabilities[c])[int(predicted_y)]
        c += 1
        for j in range(l, l + window_size):
            a = []
            for _ in range(samples):
                prob_i = softmax(all_probabilities[c])[int(predicted_y)]
                if len(prob_distr) > 0:
                    a.append(prob_i * prob_distr[c])
                else:
                    a.append(prob_i)
                c += 1
            if len(prob_distr) > 0:
                pred_probs[j].append(we(sum(a), window_prob))
            else:
                pred_probs[j].append(we(np.average(a), window_prob))

    weight_evidence = []
    for w in range(n):
        weight_evidence.append(np.average(pred_probs[w]))
    stacked = np.column_stack((split, weight_evidence))
    # print(stacked)
    print(colour_sentence(stacked))
    # print(prob_y)
    print('Predicted class label:', predicted_y)


def explain_couples(sentence, mutation_method, samples=400, removed_mode=False):
    split = sentence.split()
    n = len(split)
    x_v = []
    couples = []
    for l in range(n - 1):
        for j in range(l + 1, n):
            if removed_mode:
                s = list(split)
                del s[j]
                del s[l]
                x_v.append(' '.join(s))
            else:
                rep1 = mutation_method(sentence, l)
                rep2 = mutation_method(sentence, j)
                for i in range(len(rep1)):
                    s = rep1[i].split()
                    s[j] = rep2[i].split()[j]
                    x_v.append(' '.join(s))
            couples.append((split[l], split[j]))
    x_v.append(sentence)

    all_probabilities, all_predictions = predict(x_v)
    predicted_y = all_predictions[-1]
    prob_y = softmax(all_probabilities[-1])[int(predicted_y)]

    pred_probs = [[] for _ in range(len(couples))]

    for c in range(len(couples)):
        for i in range(samples * c, samples * (c + 1)):
            pred_probs[c].append(softmax(all_probabilities[i])[int(predicted_y)])
    weight_evidence = []
    for w in range(len(couples)):
        weight_evidence.append(np.average(pred_probs[w]))
    stacked = np.column_stack((couples, weight_evidence))
    stacked = list(sorted(stacked, key=lambda x: float(x[2])))
    print(stacked)
    print(prob_y)
    print('Predicted class label:', predicted_y)


# -elling , portrayed with quiet fastidiousness by per christian ellefsen , is a truly singular character , one whose frailties are only slightly magnified versions of the ones that vex nearly everyone .

# with open('data/rt-polaritydata/rt-polarity.pos', 'r') as f:
#     for line in f:
sentence = 'i loved watching walken in this role thinking about what the future held for him . '
# sentence = 'the storyplot was okay by itself , but the film felt very bubbly and fake .'
print(sentence)

# S = 400
# print('unigram')
# explain_couples(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1))
# print('bigram')
# explain_couples(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=2))
# print('trigram')
# explain_couples(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=3))
# print('cbow')
# explain_couples(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1, window_mode='cbow', cbow_most_prob=False))
# print('removed')
# explain_couples(sentence, new_continuous_data.get_removed_mutations, 1, True)
# print('unked')
# explain_couples(sentence, new_continuous_data.get_unked_mutations, 1)
S = 10
print('SLIDING WINDOW')
print('cbow')
explain_window(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1, window_mode='cbow', cbow_most_prob=False), S, use_prob_distr=True)
print('unigram')
explain_window(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1), S)
print('bigram')
explain_window(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=2), S)
print('trigram')
explain_window(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=3), S)
print('removed')
explain_window(sentence, new_continuous_data.get_removed_mutations, 1)
print('unked')
explain_window(sentence, new_continuous_data.get_unked_mutations, 1)

S = 5000
print('NO SLIDING WINDOW')
print('cbow')
explain(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1, window_mode='cbow', cbow_most_prob=False))
print('unigram')
explain(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=1))
print('bigram')
explain(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=2))
print('trigram')
explain(sentence, lambda x, index: new_continuous_data.get_mutations(x, S, index, window_size=3))
print('removed')
explain(sentence, new_continuous_data.get_removed_mutations)
print('unked')
explain(sentence, new_continuous_data.get_unked_mutations)