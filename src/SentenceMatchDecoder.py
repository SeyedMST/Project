# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import SentenceMatchTrainer
from SentenceMatchModelGraph import SentenceMatchModelGraph

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re



tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts








def show_weights(dataStream, valid_graph, sess, outpath=None,
             label_vocab=None, mode='trec', char_vocab=None, POS_vocab=None, NER_vocab=None, flag_valid=False,
             word_vocab=None
             ,show_attention=True):
    # if outpath is not None: outfile = open(outpath, 'wt')
    # subfile = ''
    # goldfile = ''
    # if FLAGS.is_answer_selection == True:
    # print ('open')
    #    outpath = '../trec_eval-8.0/'
    #    subfile = open(outpath + 'submission.txt', 'wt')
    #    goldfile = open(outpath + 'gold.txt', 'wt')
    # total_tags = 0.0
    # correct_tags = 0.0
    dataStream.reset()
    # last_trec = ""
    # id_trec = 0
    # doc_id_trec = 1
    # sub_list = []
    # has_true_label = set ()
    scores = []
    labels = []
    sent1s = []  # to print test sentence result
    sent2s = []  # to print test sentence result
    atts = []  # to print attention weights
    for batch_index in range(dataStream.get_num_batch()):
        cur_dev_batch = dataStream.get_batch(batch_index)
        (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
         char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch,
         sent1_char_length_batch, sent2_char_length_batch,
         POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch) = cur_dev_batch
        feed_dict = {
            valid_graph.get_truth(): label_id_batch,
            valid_graph.get_question_lengths(): sent1_length_batch,
            valid_graph.get_passage_lengths(): sent2_length_batch,
            valid_graph.get_in_question_words(): word_idx_1_batch,
            valid_graph.get_in_passage_words(): word_idx_2_batch,
            valid_graph.get_overlap(): overlap_batch,
        }

        feed_dict[valid_graph.get_question_count()] = 0
        feed_dict[valid_graph.get_answer_count()] = 0


        if show_attention == True:
            att_scores = sess.run(valid_graph.get_attention_weights(), feed_dict=feed_dict)
            for j in range (len (att_scores)):
                s1 = re.split('\\s+', sent1_batch[j])
                s2 = re.split('\\s+', sent2_batch[j])
                print (sent1_batch[j])
                print (sent2_batch[j])
                print (att_scores [j])
                fig, ax = plt.subplots()

                im, cbar = heatmap(np.transpose(att_scores [j]), s1, s2, ax=ax,
                                   cmap="YlGn", cbarlabel="attetnion score")
                #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                fig.tight_layout()
                plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, default= "../in_path.txt", help='the path to the test file.')
    #parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default="../data/glove/my_glove.840B.300d.txt",
                        help='word embedding file for the input file.')
    #parser.add_argument('--mode', type=str, default="prediction", help='prediction or probs')
    parser.add_argument('--is_trec', type=bool, default=False, help='prediction or probs')
    parser.add_argument('--index', type=str, default='29', help='prediction or probs')



    args, unparsed = parser.parse_known_args()

    op = 'wik'
    if args.is_trec == True or args.is_trec == 'True':
        op = 'tre'
    log_dir = '../models' + op
    path_prefix = log_dir + "/SentenceMatch.normal"

    #model_prefix = args.model_prefix
    in_path = args.in_path
    word_vec_path = args.word_vec_path
    out_json_path = None
    dump_prob_path = None
    
    # load the configuration file
    print('Loading configurations.')
    FLAGS = namespace_utils.load_namespace(path_prefix + args.index+ ".config.json")
    print(FLAGS)

    with_POS=False
    if hasattr(FLAGS, 'with_POS'): with_POS = FLAGS.with_POS
    with_NER=False
    if hasattr(FLAGS, 'with_NER'): with_NER = FLAGS.with_NER
    wo_char = False
    if hasattr(FLAGS, 'wo_char'): wo_char = FLAGS.wo_char

    wo_left_match = False
    if hasattr(FLAGS, 'wo_left_match'): wo_left_match = FLAGS.wo_left_match

    wo_right_match = False
    if hasattr(FLAGS, 'wo_right_match'): wo_right_match = FLAGS.wo_right_match

    wo_full_match = False
    if hasattr(FLAGS, 'wo_full_match'): wo_full_match = FLAGS.wo_full_match

    wo_maxpool_match = False
    if hasattr(FLAGS, 'wo_maxpool_match'): wo_maxpool_match = FLAGS.wo_maxpool_match

    wo_attentive_match = False
    if hasattr(FLAGS, 'wo_attentive_match'): wo_attentive_match = FLAGS.wo_attentive_match

    wo_max_attentive_match = False
    if hasattr(FLAGS, 'wo_max_attentive_match'): wo_max_attentive_match = FLAGS.wo_max_attentive_match


    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(word_vec_path, fileformat='txt3')
    label_vocab = Vocab(path_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    
    POS_vocab = None
    NER_vocab = None
    char_vocab = None
    # if with_POS: POS_vocab = Vocab(path + ".POS_vocab", fileformat='txt2')
    # if with_NER: NER_vocab = Vocab(model_prefix + ".NER_vocab", fileformat='txt2')
    char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
    #print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchTrainer.SentenceMatchDataStream(in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=1, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=20, max_sent_length=100,
                                                  is_as=True, is_word_overlap=False,
                                                  is_lemma_overlap= False)


    #print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    #print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

    if wo_char: char_vocab = None
    init_scale = 0.01
    best_path = path_prefix + args.index + ".best.model"
    print('Decoding on the test set:')
    with tf.Graph().as_default():
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchTrainer.SentenceMatchModelGraph(num_classes=2, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                  dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False,
                                                  MP_dim=FLAGS.MP_dim,
                                                  context_layer_num=FLAGS.context_layer_num,
                                                  aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                  fix_word_vec=FLAGS.fix_word_vec,
                                                  with_filter_layer=FLAGS.with_filter_layer,
                                                  with_input_highway=FLAGS.with_highway,
                                                  word_level_MP_dim=FLAGS.word_level_MP_dim,
                                                  with_match_highway=FLAGS.with_match_highway,
                                                  with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                  highway_layer_num=FLAGS.highway_layer_num,
                                                  with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                  lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                  with_left_match=(not FLAGS.wo_left_match),
                                                  with_right_match=(not FLAGS.wo_right_match),
                                                  with_full_match=(not FLAGS.wo_full_match),
                                                  with_maxpool_match=(not FLAGS.wo_maxpool_match),
                                                  with_attentive_match=(not FLAGS.wo_attentive_match),
                                                  with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
                                                  with_bilinear_att=(FLAGS.attention_type)
                                                  , type1=FLAGS.type1, type2=FLAGS.type2, type3=FLAGS.type3,
                                                  with_aggregation_attention=not FLAGS.wo_agg_self_att,
                                                  is_answer_selection=FLAGS.is_answer_selection,
                                                  is_shared_attention=FLAGS.is_shared_attention,
                                                  modify_loss=FLAGS.modify_loss,
                                                  is_aggregation_lstm=FLAGS.is_aggregation_lstm,
                                                  max_window_size=FLAGS.max_window_size
                                                  , prediction_mode=FLAGS.prediction_mode,
                                                  context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
                                                  is_aggregation_siamese=FLAGS.is_aggregation_siamese
                                                  , unstack_cnn=FLAGS.unstack_cnn,
                                                  with_context_self_attention=FLAGS.with_context_self_attention,
                                                  mean_max=FLAGS.mean_max, clip_attention=FLAGS.clip_attention
                                                  , with_tanh=FLAGS.tanh, new_list_wise=FLAGS.new_list_wise,
                                                  q_count=1, pos_avg=FLAGS.pos_avg,
                                                  with_input_embedding=FLAGS.with_input_embedding
                                                  , with_output_highway=FLAGS.with_output_highway,
                                                  with_matching_layer=FLAGS.with_matching_layer)
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
                
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)



        show_weights(testDataStream, valid_graph, sess, outpath='', label_vocab=label_vocab,mode='',
                                                 char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab
                     ,show_attention=True)


