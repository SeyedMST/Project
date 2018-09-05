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


#heatmap(att_scores [j], s2, s1, ax=ax,
#                                       cmap="YlGn", cbarlabel="attetnion score")

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
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
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
                if label_id_batch [j] == 1:
                    print (sent1_batch[j])
                    print (sent2_batch[j])
                    print (att_scores [j])
                    fig, ax = plt.subplots()

                    im, cbar = heatmap(att_scores [j], s2, s1, ax=ax,
                                       cmap="YlGn", cbarlabel="attetnion score")
                    #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                    fig.tight_layout()
                    plt.show()





if __name__ == '__main__':

    epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    pos_trec = [352.0, 191.0, 130.0, 95.0, 60.0, 45.0, 32.0, 32.0, 33.0, 18.0, 20.0, 23.0, 13.0, 15.0, 10.0, 13.0, 4.0, 10.0, 9.0,
     4.0, 6.0, 3.0, 5.0, 2.0, 6.0, 1.0, 4.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 3.0, 2.0, 3.0, 1.0, 2.0, 0.0, 3.0, 1.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    pos_wiki = [734.0, 98.0, 20.0, 1.0, 1.0, 1.0, 2.0]

    #map_listpl = [4759762.387, 4751877.19, 4749028.117, 4742046.493, 4736566.713, 4731218.37, 4730733.15, 4725699.15, 4768785.58, 4567046.867]
    #map_listmle = [4754315.949, 4766392.034, 4697314.583, 4730979.343, 4691711.086, 4668025.715, 4680640.321, 4652065.742, 4652879.819, 4533239.929]

    map_listpl = [159.615, 159.052, 158.588, 158.445, 158.137, 157.653, 157.196, 156.845, 156.891, 156.741]
    map_listmle = [159.61, 157.709, 156.783, 155.864, 154.947, 154.168, 153.421, 152.578, 151.921, 151.237]


    epoch = []
    for i in range (len(pos_trec)):
        epoch.append(i + 1)
    while len(pos_wiki) < len(pos_trec):
        pos_wiki.append(0)
    plt.plot (epoch, pos_trec, label = 'QASent', marker = None, color = 'r', linestyle=None)
    plt.plot (epoch, pos_wiki, label = 'WikiQA', marker = None, color = 'b', linestyle='-.')

    plt.xlabel('Positive Answers Count')
    plt.ylabel('Question Count')
    plt.legend()
    plt.show()

    epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #TREC:

    # map_listnet =[0.730, 0.740, 0.780, 0.784, 0.797, 0.802, 0.818, 0.812, 0.821, 0.819]
    # map_listmle =[0.704, 0.721, 0.757, 0.664, 0.633, 0.660, 0.622, 0.648, 0.616, 0.630]
    # map_zero = [0.740, 0.751, 0.792, 0.795, 0.801, 0.817, 0.824, 0.842, 0.855, 0.840]
    # map_lambmle = [0.742, 0.775, 0.801, 0.819, 0.826, 0.827, 0.839, 0.859, 0.848, 0.846]
    # map_listpl = [0.644, 0.696, 0.711, 0.698, 0.755, 0.763, 0.776, 0.776, 0.798, 0.793]

    #train:
    # map_lambmle = [0.816, 0.861, 0.882, 0.893, 0.914, 0.922, 0.933, 0.946, 0.954, 0.965]
    # map_zero = [0.812, 0.849, 0.866, 0.875, 0.889, 0.904, 0.907, 0.923, 0.932, 0.935]
    # map_listnet = [0.76, 0.821, 0.849, 0.849, 0.885, 0.894, 0.901, 0.914, 0.921, 0.926]
    # map_pointwise = [0.776, 0.802, 0.827, 0.844, 0.857, 0.867, 0.875, 0.891, 0.897, 0.905]
    # map_listmle = [0.634, 0.632, 0.631, 0.585, 0.582, 0.595, 0.559, 0.567, 0.538, 0.59]
    # map_listpl = [0.686, 0.75, 0.782, 0.789, 0.815, 0.823, 0.831, 0.84, 0.848, 0.848]

    map_lambmle = [0.81, 0.852, 0.877, 0.895, 0.901, 0.92, 0.924, 0.936, 0.95, 0.955]
    map_zero = [0.802, 0.858, 0.871, 0.88, 0.89, 0.901, 0.913, 0.925, 0.931, 0.939]
    map_listnet = [0.779, 0.804, 0.838, 0.861, 0.874, 0.887, 0.89, 0.894, 0.901, 0.906]
    map_pointwise = [0.777, 0.807, 0.82, 0.83, 0.839, 0.849, 0.86, 0.865, 0.862, 0.875]
    #map_listpl = [0.716, 0.738, 0.729, 0.784, 0.803, 0.797, 0.825, 0.826, 0.832, 0.841]
    #map_listmle = [0.668, 0.682, 0.681, 0.69, 0.657, 0.619, 0.617, 0.614, 0.584, 0.587]
    map_listpl = [0.713, 0.77, 0.82, 0.831, 0.84, 0.853, 0.862, 0.87, 0.875, 0.88]
    map_listmle = [0.715, 0.762, 0.753, 0.742, 0.755, 0.754, 0.744, 0.78, 0.771, 0.784]
    #
    # #
    # #
    #
    # #WIKI:
    # map_lambmle = [0.73, 0.794, 0.856, 0.907, 0.945, 0.963, 0.968, 0.979, 0.986, 0.989]
    #               #[0.729, 0.784, 0.851, 0.905, 0.949, 0.964, 0.966, 0.98, 0.988, 0.986]
    # map_zero = [0.720, 0.786, 0.855, 0.899, 0.932, 0.951, 0.967, 0.973, 0.981, 0.981]
    #            #[0.735, 0.808, 0.863, 0.916, 0.942, 0.954, 0.974, 0.978, 0.984, 0.98]
    # map_listnet = [0.705, 0.756, 0.823, 0.878, 0.91, 0.936, 0.952, 0.966, 0.968, 0.975]
    # map_pointwise = [0.671, 0.715, 0.762, 0.813, 0.837, 0.866, 0.895, 0.921, 0.935, 0.944]
    # #map_listpl = [0.587, 0.604, 0.62, 0.642, 0.673, 0.692, 0.699, 0.712, 0.728, 0.754]
    # #map_listmle = [0.625, 0.63, 0.664, 0.651, 0.677, 0.681, 0.663, 0.687, 0.668, 0.698]
    # map_listmle = [0.639, 0.64, 0.656, 0.669, 0.664, 0.685, 0.702, 0.71, 0.722, 0.722]
    # map_listpl = [0.643, 0.646, 0.661, 0.668, 0.679, 0.689, 0.687, 0.698, 0.701, 0.708]
    #
    # plt.plot (epoch, map_lambmle, label = 'LambMLE', marker = 's', color = 'r', linestyle='-')
    # plt.plot (epoch, map_zero, label = 'ZeroListNet', marker = 's', color = 'b', linestyle='-')
    # plt.plot (epoch, map_listnet, label = 'ListNet', marker = 's', color = 'g', linestyle='-')
    # #plt.plot (epoch, map_pointwise, label = 'Pointwise', marker = 's', color = 'p', linestyle='-')
    plt.plot (epoch, map_listpl, label = 'ListPL', marker = 's', color = 'm', linestyle='-')
    plt.plot (epoch, map_listmle, label = 'ListMLE', marker = 's', color = 'y', linestyle='-')
    #
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, default= "../in_path.txt", help='the path to the test file.')
    #parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, default="../data/glove/my_glove.840B.300d.txt",
                        help='word embedding file for the input file.')
    #parser.add_argument('--mode', type=str, default="prediction", help='prediction or probs')
    parser.add_argument('--is_trec', type=bool, default=True, help='prediction or probs')
    parser.add_argument('--index', type=str, default='64', help='prediction or probs') #29 # 'we1-1' #64




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

    devDataStream = SentenceMatchTrainer.SentenceMatchDataStream("../dev.txt", word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=10, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=20, max_sent_length=100,
                                                  is_as=True, is_word_overlap=False,
                                                  is_lemma_overlap= False)

    testDataStream = SentenceMatchTrainer.SentenceMatchDataStream(in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=10, isShuffle=False, isLoop=True, isSort=True,
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
            if "aggregation_layer" in var.name:
                print (var.name)
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)



        # show_weights(testDataStream, valid_graph, sess, outpath='', label_vocab=label_vocab,mode='',
        #                                          char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab
        #              ,show_attention=True)

        #95
        for i in range (95, 96):
            char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
            devDataStream = SentenceMatchTrainer.SentenceMatchDataStream("../dev.txt", word_vocab=word_vocab,
                                                                         char_vocab=char_vocab,
                                                                         POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                                         label_vocab=label_vocab,
                                                                         batch_size=i, isShuffle=False, isLoop=True,
                                                                         isSort=True,
                                                                         max_char_per_word=20, max_sent_length=100,
                                                                         is_as=True, is_word_overlap=False,
                                                                         is_lemma_overlap=False)

            testDataStream = SentenceMatchTrainer.SentenceMatchDataStream(in_path, word_vocab=word_vocab,
                                                                          char_vocab=char_vocab,
                                                                          POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                                          label_vocab=label_vocab,
                                                                          batch_size=i, isShuffle=False, isLoop=True,
                                                                          isSort=True,
                                                                          max_char_per_word=20, max_sent_length=100,
                                                                          is_as=True, is_word_overlap=False,
                                                                          is_lemma_overlap=False)
            char_vocab = None
            my_map, my_mrr = SentenceMatchTrainer.evaluate(devDataStream, valid_graph, sess, char_vocab=char_vocab,
                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab)
            print ("*****************************************")
            print (my_map, my_mrr)
            my_map, my_mrr = SentenceMatchTrainer.evaluate(testDataStream, valid_graph, sess, char_vocab=char_vocab,
                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab)
            print (my_map, my_mrr)
