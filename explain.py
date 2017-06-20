import tensorflow as tf
import os
from tensorflow.contrib import learn
import data_helpers
import numpy as np
import csv
import new_data
import math

def softmax(scores):
    s1, s2 = scores[0], scores[1]
    e1 =  math.exp(s1)
    e2 = math.exp(s2)
    return e1/(e1+e2), e2/(e1+e2)

def explain(sentence):
    # Parameters
    # ==================================================

    # Eval Parameters
    tf.flags.DEFINE_string("checkpoint_dir", "./runs/1497534303/checkpoints/", "Checkpoint directory from training run")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

    S = 100

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    x_v = new_data.get_S_ngram_mutations(sentence, S, 'unigram')
    x_v.append(sentence)

    x_variants = np.array(list(vocab_processor.transform(x_v)))
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

            #TODO: GET THE NEW LIST
            batches = data_helpers.batch_iter(list(x_variants), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = []

            for x_test_batch in batches:
                results = sess.run([predictions, probabilities], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, results[0]])
                for result in results[1]:
                    all_probabilities.append(result)
    predicted_y = all_predictions[-1]
    print(predicted_y)
    prob_y = softmax(all_probabilities[-1])[int(predicted_y)]

    words = sentence.split()
    weight_evidence = []

    for i in range(len(words)):
        pred_probs = []
        for j in range(S * i, S * (i + 1)):
            print(x_v[j])
            print(softmax(all_probabilities[j])[int(predicted_y)])
            pred_probs.append(softmax(all_probabilities[j])[int(predicted_y)])
        weight_evidence.append(np.average(pred_probs) - prob_y)
    stacked = np.column_stack((words, weight_evidence))
    print(stacked)
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "explain.csv")
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(stacked)
    print(prob_y)
    print('Predicted class label:', predicted_y)
    print(checkpoint_file)

explain('once again mr . Costner has dragged out a movie for far longer than necessary . aside from the terrific sea rescue sequences , of which there are very few i just did not care about any of the characters . most of us have ghosts in the closet , and Costner\'s character are realized early on , and then forgotten until much later , by which time I did not care . the character we should really care about is a very cocky , overconfident Ashton Kutcher . the problem is he comes off as kid who thinks he\'s better than anyone else around him and shows no signs of a cluttered closet . '
        'his only obstacle appears to be winning over Costner . finally when we are well past the half way point of this stinker , Costner tells us all about Kutcher\'s ghosts . we are told why Kutcher is driven to be the best with no prior inkling or foreshadowing . no magic here, it was all I could do to keep from turning it off an hour in s.')