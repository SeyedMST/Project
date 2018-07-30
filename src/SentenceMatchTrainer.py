# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
#import pandas as pd
import subprocess
import random
import numpy as np

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils



def sort_mle (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                                             char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch,
                                             sent1_char_length_batch, sent2_char_length_batch,
                                             POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch):
    l = []
    for i in range(len(label_id_batch)):
        l.append((label_batch[i], sent1_batch[i], sent2_batch[i], label_id_batch[i], word_idx_1_batch[i], word_idx_2_batch[i],
                  sent1_length_batch[i], sent2_length_batch[i],
                                             overlap_batch[i]))
    random.shuffle(l)
    l = sorted(l, key=lambda instance: (instance[3]), reverse=True)  # sort based on len (answer[i])
    label_batch = []
    sent1_batch = []
    sent2_batch = []
    label_id_batch = []
    word_idx_1_batch = []
    word_idx_2_batch = []
    #char_matrix_idx_1_batch = []
    #char_matrix_idx_2_batch = []
    sent1_length_batch = []
    sent2_length_batch = []
    #sent1_char_length_batch = []
    #sent2_char_length_batch = []
    overlap_batch = []
    for i in range(len(l)):
        label_batch.append (l[i][0])
        sent1_batch.append(l[i][1])
        sent2_batch.append(l[i][2])
        label_id_batch.append(l[i][3])
        word_idx_1_batch.append(l[i][4])
        word_idx_2_batch.append(l[i][5])
        sent1_length_batch.append(l[i][6])
        sent2_length_batch.append(l[i][7])
        overlap_batch.append(l[i][8])

    label_batch = np.array (label_batch)
    sent1_batch = np.array(sent1_batch)
    sent2_batch = np.array(sent2_batch)
    label_id_batch = np.array(label_id_batch)
    word_idx_1_batch = np.array (word_idx_1_batch)
    word_idx_2_batch = np.array (word_idx_2_batch)
    sent1_length_batch = np.array(sent1_length_batch)
    sent2_length_batch = np.array(sent2_length_batch)
    overlap_batch = np.array (overlap_batch)
    return (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                                             char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch,
                                             sent1_char_length_batch, sent2_char_length_batch,
                                             POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch)




import nltk
eps = 1e-8
FLAGS = None

def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt')
    for line in infile:
        if sys.version_info[0] < 3:
            line = line.decode('utf-8').strip()
        else:
            line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[2]
        sentence1 = re.split("\\s+",items[0].lower())
        sentence2 = re.split("\\s+",items[1].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def evaluate(dataStream, valid_graph, sess, outpath=None,
             label_vocab=None, mode='trec',char_vocab=None, POS_vocab=None, NER_vocab=None, flag_valid = False,word_vocab = None
             ,first_on_best_model = False):
    outpath = ''
    #if outpath is not None: outfile = open(outpath, 'wt')
    #subfile = ''
    #goldfile = ''
    #if FLAGS.is_answer_selection == True:
        #print ('open')
    #    outpath = '../trec_eval-8.0/'
    #    subfile = open(outpath + 'submission.txt', 'wt')
    #    goldfile = open(outpath + 'gold.txt', 'wt')
    #total_tags = 0.0
    #correct_tags = 0.0
    dataStream.reset()
    #last_trec = ""
    #id_trec = 0
    #doc_id_trec = 1
    #sub_list = []
    #has_true_label = set ()
    questions_count = 0.0
    MAP = 0.0
    MRR = 0.0
    scores = []
    labels = []
    sent1s = [] # to print test sentence result
    sent2s = [] # to print test sentence result
    atts = [] # to print attention weights
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

                    valid_graph.get_overlap():overlap_batch,
#                     valid_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                     valid_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                     valid_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                     valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                }

        if char_vocab is not None and FLAGS.wo_char == False:
            feed_dict[valid_graph.get_question_char_lengths()] = sent1_char_length_batch
            feed_dict[valid_graph.get_passage_char_lengths()] = sent2_char_length_batch
            feed_dict[valid_graph.get_in_question_chars()] = char_matrix_idx_1_batch
            feed_dict[valid_graph.get_in_passage_chars()] = char_matrix_idx_2_batch

        if POS_vocab is not None:
            feed_dict[valid_graph.get_in_question_poss()] = POS_idx_1_batch
            feed_dict[valid_graph.get_in_passage_poss()] = POS_idx_2_batch

        if NER_vocab is not None:
            feed_dict[valid_graph.get_in_question_ners()] = NER_idx_1_batch
            feed_dict[valid_graph.get_in_passage_ners()] = NER_idx_2_batch

        if FLAGS.is_answer_selection == True:
            feed_dict[valid_graph.get_question_count()] = 0#dataStream.question_count(batch_index)
            feed_dict[valid_graph.get_answer_count()] = 0#dataStream.answer_count(batch_index)

        #total_tags += len(label_batch)
        #correct_tags += sess.run(valid_graph.get_eval_correct(), feed_dict=feed_dict)
        if outpath is not None: # hich vaght None nist khate aval meghdar dadam behesh :D
            #if mode =='prediction':
            #    predictions = sess.run(valid_graph.get_predictions(), feed_dict=feed_dict)
            #    for i in xrange(len(label_batch)):
            #        outline = label_batch[i] + "\t" + label_vocab.getWord(predictions[i]) + "\t" + sent1_batch[i] + "\t" + sent2_batch[i] + "\n"
            #        outfile.write(outline.encode('utf-8'))

            if FLAGS.is_answer_selection == True:
                scores.append(sess.run(valid_graph.get_score(), feed_dict=feed_dict))
                labels.append (label_id_batch)
                if flag_valid == True or first_on_best_model == True:
                    sent1s.append(sent1_batch)
                    sent2s.append(sent2_batch)
                    if FLAGS.store_att == True and first_on_best_model == False:
                        atts.extend(np.split(sess.run(valid_graph.get_attention_weights(), feed_dict=feed_dict),len(sent1_batch)))

                # for i in xrange(len(label_batch)):
                #     if sent1_batch[i] != last_trec:
                #         last_trec = sent1_batch[i]
                #         id_trec += 1
                #     if (FLAGS.prediction_mode == 'point_wise'):
                #         pbi = ouput_prob1(probs[i], label_vocab, '1')
                #     else:
                #         pbi = probs[i]
                #     if (label_batch[i] == '1'):
                #         has_true_label.add(id_trec)
                #     sub_list.append((id_trec, doc_id_trec, pbi, label_batch[i]))
                #     doc_id_trec +=1
            #else:
                #probs = sess.run(valid_graph.get_prob(), feed_dict=feed_dict)
                #for i in xrange(len(label_batch)):
                #    outfile.write(label_batch[i] + "\t" + output_probs(probs[i], label_vocab) + "\n")
    #print ('start')

    #if FLAGS.is_answer_selection == False:
    #    if outpath is not None: outfile.close()

    if FLAGS.is_answer_selection == True:
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        if flag_valid == True or first_on_best_model == True:
            sent1s = np.concatenate(sent1s)
            sent2s = np.concatenate(sent2s)
            #atts = np.concatenate(atts)
        return MAP_MRR(scores, labels, dataStream.get_candidate_answer_length(), flag_valid
                                   ,sent1s, sent2s, atts ,word_vocab, first_on_best_model)
        # print (final_map, final_mrr)
        # for i in xrange(len (sub_list)):
        #     id_trec, doc_id_trec, prob1, label_gold = sub_list[i]
        #     if id_trec in has_true_label:
        #         subfile.write(str(id_trec) + " 0 " + str(doc_id_trec)
        #                       + " 0 " + str(prob1) + ' nnet\n')
        #         goldfile.write(str(id_trec) + " 0 " + str(doc_id_trec)
        #                        + " " + label_gold + '\n')
        # subfile.close()
        # goldfile.close()
        #print ('hi')
        # p = subprocess.check_output("/bin/sh ../run_eval.sh '{}'".format(outpath),
        #                 shell=True)
        #print (p)
        # p = p.split()
        #my_map = float(p[2])
        #my_mrr = float(p[5])

        #print("map '{}' , mrr '{}'".format(my_map, my_mrr))

    #    print ('end')
    #accuracy = correct_tags / total_tags * 100
    #return accuracy


def MAP_MRR(logit, gold, candidate_answer_length, flag_valid, sent1s, sent2s, atts, word_vocab
            ,first_on_best_model):
    c_1_j = 0.0 #map
    c_2_j = 0.0 #mrr
    visited = 0
    output_sentences = []
    output_attention_weights = []
    for i in range(len(candidate_answer_length)):
        prob = logit[visited: visited + candidate_answer_length[i]]
        label = gold[visited: visited + candidate_answer_length[i]]
        if flag_valid == True or first_on_best_model == True:
            question = sent1s[visited: visited + candidate_answer_length[i]]
            answers = sent2s[visited: visited + candidate_answer_length[i]]
            if FLAGS.store_att == True and first_on_best_model == False:
                attention_weights = atts [visited: visited + candidate_answer_length[i]]
        visited += candidate_answer_length[i]
        rank_index = np.argsort(prob).tolist()
        rank_index = list(reversed(rank_index))
        score = 0.0
        count = 0.0
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                count += 1
                score += count / i
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                c_2_j += 1 / float(i)
                break
        c_1_j += score / count

        if flag_valid == True:
            output_sentences.append(word_vocab.to_word_in_sequence(question[0]) + "\n")
            for jj in range(len(answers)):
                output_sentences.append(str(label[rank_index[jj]]) + " " + str(prob[rank_index[jj]]) + "- " +
                                        word_vocab.to_word_in_sequence(answers[rank_index[jj]]) + "\n")
                if FLAGS.store_att == True:
                    output_attention_weights.append(str (attention_weights[rank_index[jj]]) + '\n')
            output_sentences.append("AP: {} \n\n".format(score/count))
        if first_on_best_model == True:
            #output_sentences.append(word_vocab.to_word_in_sequence(question[0]) + '\t')
            for jj in range(len(answers)):
                lj = int(np.ceil(label[rank_index[jj]] - eps))
                output_sentences.append(word_vocab.to_word_in_sequence(question[0]) + '\t' +
                                        word_vocab.to_word_in_sequence(answers[rank_index[jj]]) + '\t' +
                                        str(lj) + '\n')


    my_map = c_1_j/len(candidate_answer_length)
    my_mrr = c_2_j/len(candidate_answer_length)
    if flag_valid == False and first_on_best_model == False:
        return (my_map,my_mrr)
    else:
        return (my_map, my_mrr, output_sentences, output_attention_weights)

def ouput_prob1(probs, label_vocab, lable_true):
    out_string = ""
    for i in range(probs.size):
        if label_vocab.getWord(i) == lable_true:
            return probs[i]

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()


def Generate_random_initialization(cnf):
    if FLAGS.is_random_init == True:
        # cnf = cnf % 4
        # FLAGS.cnf = cnf
        # type1 = ['w_mul']
        # type2 = ['w_sub',None]
        # type3 = [None]
        # FLAGS.type1 = random.choice(type1)
        # FLAGS.type2 = random.choice(type2)
        # FLAGS.type3 = random.choice(type3)
        # context_layer_num = [1]
        aggregation_layer_num = [1]
        FLAGS.aggregation_layer_num = random.choice(aggregation_layer_num)
        # FLAGS.context_layer_num = random.choice(context_layer_num)
        # #if cnf == 1  or cnf == 4:
        # #    is_aggregation_lstm = [True]
        # #elif cnf == 2:
        # #    is_aggregation_lstm =  [False]
        # #else: #3
        # #is_aggregation_lstm = [True]#[True, False]
        # FLAGS.is_aggregation_lstm = True#random.choice(is_aggregation_lstm)
        # # max_window_size = [1] #[x for x in range (1, 4, 1)]
        # # FLAGS.max_window_size = random.choice(max_window_size)
        # #
        # # att_cnt = 0
        # # if FLAGS.type1 != None:
        # #     att_cnt += 1
        # # if FLAGS.type2 != None:
        # #     att_cnt += 1
        # # if FLAGS.type3 != None:
        # #     att_cnt += 1
        #
        #
        # #context_lstm_dim:
        if FLAGS.context_layer_num == 2:
             context_lstm_dim = [50] #[x for x in range(50, 110, 10)]
        else:
             context_lstm_dim = [150]#[150, 200]#[x for x in range(50, 160, 10)]
        #
        if FLAGS.aggregation_layer_num == 2:
            aggregation_lstm_dim = [50]#[x for x in range (50, 110, 10)]
        else:
            aggregation_lstm_dim = [70]#[70, 100]#[x for x in range (50, 160, 10)]
        # # else: # CNN
        # #     if FLAGS.max_window_size == 1:
        # #         aggregation_lstm_dim = [100]#[x for x in range (50, 801, 10)]
        # #     elif FLAGS.max_window_size == 2:
        # #         aggregation_lstm_dim = [100]#[x for x in range (50, 510, 10)]
        # #     elif FLAGS.max_window_size == 3:
        # #         aggregation_lstm_dim = [50]#[x for x in range (50, 410, 10)]
        # #     elif FLAGS.max_window_size == 4:
        # #         aggregation_lstm_dim = [x for x in range (50, 210, 10)]
        # #     else: #5
        # #         aggregation_lstm_dim = [x for x in range (50, 110, 10)]
        #
        #
        MP_dim = [70]#[30, 50, 70]#[20,50,100]#[x for x in range (20, 610, 10)]
        # learning_rate = [0.002]#[0.001, 0.002, 0.003, 0.004]
        dropout_rate = [0.25]#[0.1, 0.2, 0.25]#[x/100.0 for x in xrange (2, 30, 2)]
        question_count_per_batch = [4]

        # char_lstm_dim = [80] #[x for x in range(40, 110, 10)]
        # char_emb_dim = [40] #[x for x in range (20, 110, 10)]
        # wo_char = [True]
        # is_shared_attention = [True, False]#[False, True]
        # is_aggregation_siamese = [False, True]
        # clip_attention = [True]
        # mean_max = [True]
        # word_overlap = [True]
        # lemma_overlap = [True]

        #batch_size = [x for x in range (30, 80, 10)] we can not determine batch_size here





        # ************************ # we dont need tuning below parameters any more :
        #
        # wo_lstm_drop_out = [True]
        # if cnf == 10:
        #     wo_agg_self_att = [True]
        # else:
        #     wo_agg_self_att = [True]
        # #if cnf == 1:
        # attention_type = ['dot_product']#['bilinear', 'linear', 'linear_p_bias', 'dot_product']
        # #else:
        # #attention_type = ['bilinear']
        # # if cnf == 20:
        # #     with_context_self_attention = [False, True]
        # # else:
        # #     with_context_self_attention = [False]
        #
        # with_context_self_attention = [False]
        #modify_loss = [0, 0.1]#[x/10.0 for x in range (0, 5, 1)]
        #prediction_mode = ['list_wise'] #, 'list_wise', 'hinge_wise']
        #new_list_wise = [True, False]
        #if cnf == 2:
        # unstack_cnn = [False]
        # #else:
        # #    unstack_cnn = [False, True]
        # with_highway = [False]
        # if FLAGS.is_aggregation_lstm == False:
        #     with_match_highway = [False]
        # else:
        #     with_match_highway = [False]
        # with_aggregation_highway = [False]
        # highway_layer_num = [1]
        # FLAGS.with_context_self_attention = random.choice(with_context_self_attention)
        # FLAGS.batch_size = random.choice(batch_size)
        # FLAGS.unstack_cnn = random.choice(unstack_cnn)
        # FLAGS.attention_type = random.choice(attention_type)
        # FLAGS.learning_rate = random.choice(learning_rate)
        FLAGS.dropout_rate = random.choice(dropout_rate)
        # FLAGS.char_lstm_dim = random.choice(char_lstm_dim)
        FLAGS.context_lstm_dim = random.choice(context_lstm_dim)
        FLAGS.aggregation_lstm_dim = random.choice(aggregation_lstm_dim)
        FLAGS.MP_dim = random.choice(MP_dim)
        FLAGS.question_count_per_batch = random.choice(question_count_per_batch)
        # FLAGS.char_emb_dim = random.choice(char_emb_dim)
        # FLAGS.with_aggregation_highway = random.choice(with_aggregation_highway)
        # FLAGS.wo_char = random.choice(wo_char)
        # FLAGS.wo_lstm_drop_out = random.choice(wo_lstm_drop_out)
        # FLAGS.wo_agg_self_att = random.choice(wo_agg_self_att)
        # FLAGS.is_shared_attention = random.choice(is_shared_attention)
        #FLAGS.modify_loss = random.choice(modify_loss)
        #FLAGS.prediction_mode = random.choice(prediction_mode)
        #FLAGS.new_list_wise = random.choice(new_list_wise)
        # FLAGS.with_match_highway = random.choice(with_match_highway)
        # FLAGS.with_highway = random.choice(with_highway)
        # FLAGS.highway_layer_num = random.choice(highway_layer_num)
        # FLAGS.is_aggregation_siamese = random.choice(is_aggregation_siamese)

        #
        # FLAGS.MP_dim = FLAGS.MP_dim // (att_cnt*FLAGS.context_layer_num)
        # FLAGS.MP_dim = (FLAGS.MP_dim+10) - FLAGS.MP_dim % 10
        #
        # if (FLAGS.type1 == 'mul' or FLAGS.type2 == 'mul' or FLAGS.type3 == 'mul'):
        #     clstm = FLAGS.context_lstm_dim
        #     mp = FLAGS.MP_dim
        #     while (clstm*2) % mp != 0:
        #         mp -= 10
        #     FLAGS.MP_dim = mp

        print (FLAGS)

    if cnf == 300:
        return False
    else:
        return True

    #return True


def Get_Next_box_size (index):
    #sort301 box_size = 150 ,govle6b.300d, differetn drop_out, ... without start batch ,... arc :)
    #list = [15, 15,  205, 205, 25, 25, 37, 37, 102, 102, 131, 131, 77, 77] #tune1- tune box size, best on 205
                                                                                #odd indices are pos_avg = False
                                                                                #here a bit better on pos_avg = True
                                                                                #here we have S for train! not test!!
    #list = [600] #tune2- box size = 600 (very less boxing!)
    #list = [120, 150, 180, 270, 450]#tre tune3-# tune box size like tune1-
    #list = [15, 15, 30, 30] #wiki tune1- wiki box size
    #list = [205, 205] #tune4- box_size = 205, drop_out = 0.1,[my_glove300d, glove300d],
                       #tune5- box_size = 205, drop_out=0.0.05, [my_glove300d, glove300d]

    #form here box_size is 700:
    #list = [15, 30, 50, 70, 100, 150, 200, 300, 10, 20, 40, 90, 110, 120] #topsample1- (batch=4, drop=0.05) just add to loss top
                                                                                     # k negative with all positives, pos_avg = True
    #list = [30, 100] #topsample2- (batch = 10) #same topsample1- with question per batch = 10 instead of 4

    #list = [1.5, 2, 3, 4,   5.0,0.1, 100.0, 1.0,1.25] #toptreshold1- just add to loss pos_avg=True thoes with
                                                    # pos - neg - marginhe < 0

    #list = [100, 100, 100, 100] #samplelist1- same topsample1- diff loss[Kl-div, pos_avg = False, Kl-div, pos_avg = True]
                                 #ablation1- same topsample1- diff types [w_mul, w_sub_self]
    #list = [100] #divcount: pos_avg = False and for each batch divide on question_count instead of pos_count, leads higher loss for
                            # batches with more positive answers and poor result.
                 #my_glove1- : tested on golve800b.300d with different configs(drop_out, ...) and
                                            #we delete drop_out ofter word embedding(input_layer) [eftezah shod!].
                                            #pos_avg = False, not same divcount but same tune1-
                 #glove1- : tested on glove6b.300d to compare with sort300. dropout ro embeding hasttttt.
    #list = [100, 100, 100] #glove2- sample_percent 100, [300, my_glove, 300sample=False]
                            #glove3- sampling = sample_100, my_glove, [pos_avg=True, pos_avg=True, kldiv]

                            #bug: too bala hame pos_avg ha True boode!

                              ##glove4- [pos_avg = True, pos_avg = False(div_count), pos_avg=False] (in hichi kolan)
                                # az inja be bad kolan divcount

    #list = [100, 100] # glove4- [pos_avg = True, False(divcount)] sadegh
    #az inja be bad listnet dorost shode
    #list = [100, 100, 100] #loss1- [point-wise, list_wise, list_wise] sadegh
    #az inja be bad sampling = False beshe
    list = [100, 100, 100] #glove5- [(glove5-0)pos_avg = True, (glove51)kl, pos_avg=True] sampling = False
                            #mle1- [30, list_net, real_list_net] wiki
    if  (index > FLAGS.end_batch):
        return False
    FLAGS.sampling = False
    FLAGS.sample_percent = list [index]
    if index == 0:
        # FLAGS.word_vec_path = "../data/glove/my_glove.840B.300d.txt"
        # FLAGS.pos_avg = True
        # FLAGS.prediction_mode = 'point_wise'
        FLAGS.word_vec_path = "../data/glove/my_glove.840B.300d.txt"
        FLAGS.pos_avg = True
        FLAGS.prediction_mode = 'list_mle'
        #FLAGS.new_list_wise = True
        FLAGS.topk = 30
    if index == 1:
        FLAGS.word_vec_path = "../data/glove/my_glove.840B.300d.txt"
        FLAGS.pos_avg = True
        FLAGS.prediction_mode = 'list_wise'
        FLAGS.new_list_wise = False
        #FLAGS.topk = 15
    if index == 2:
        #FLAGS.sampling = False
        FLAGS.word_vec_path = "../data/glove/my_glove.840B.300d.txt"
        FLAGS.pos_avg = False
        FLAGS.prediction_mode = 'real_list_net'
        #FLAGS.new_list_wise = True
        #FLAGS.topk = 10

    FLAGS.top_treshold = -1 ###list[index]

    FLAGS.max_answer_size = 1000
    FLAGS.batch_size = 1000
    FLAGS.max_epochs = 10

    FLAGS.type1 = 'w_sub_mul'
    #if index == 0:
    #    FLAGS.type1 = 'w_mul'
    #elif index == 1:
    #    FLAGS.type1 = 'w_sub_self'


    #if index == 3:
    #    FLAGS.pos_avg = True
    # if index == 0:
    #     FLAGS.pos_avg = True
    # if index == 1:
    #     FLAGS.pos_avg = False
    # else:
    #     FLAGS.new_list_wise = False
    FLAGS.sampling_type = 'attentive'
    # if list [index] < 50:
    #     FLAGS.max_epochs = 7
    # else:
    #     FLAGS.max_epochs = 8

    return True


def make_hinge_truth(truth, question_count,answer_count):
    g = truth.reshape(question_count, answer_count)
    g = np.ceil(g-eps)
    g_p = np.expand_dims(g, axis=1)
    g_n = np.expand_dims(g, axis=-1)
    mask = np.subtract(g_p, g_n)
    mask = mask + 1
    mask = mask // 2
    #print (mask)
    return mask

def get_mle_mask(le):
    mask_list = []
    for i in range(le):
        l = []
        for j in range(le):
            if i + 1 == le:
                l.append(-1.0)
            elif j < i:
                l.append(-1.0)
            elif j == i:
                l.append(1.0)
            elif j > i:
                l.append(0.0)
        mask_list.append(l)
    return np.array(mask_list)


def get_mle_mask_topk(k, l):
    mask_topk = np.zeros(l, np.float32)
    if k > l:
        k = l
    for i in range (k):
        mask_topk[i] = 1.0
    return mask_topk

#def Generate_random_box_size ():

def main(_):

    #for x in range (100):
    #    Generate_random_initialization()
    #    print (FLAGS.is_aggregation_lstm, FLAGS.context_lstm_dim, FLAGS.context_layer_num, FLAGS. aggregation_lstm_dim, FLAGS.aggregation_layer_num, FLAGS.max_window_size, FLAGS.MP_dim)

    print('Configurations:')

    if FLAGS.word_overlap == 'True':
        FLAGS.word_overlap = True

    print(FLAGS)

    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    op = 'wik'
    if FLAGS.is_trec == True or FLAGS.is_trec == 'True':
        op = 'tre'
    log_dir = FLAGS.model_dir + op
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    while (Get_Next_box_size(FLAGS.start_batch) == True):
        word_vec_path = FLAGS.word_vec_path
        word_vocab = Vocab(word_vec_path, fileformat='txt3')
        best_path = path_prefix + '.best.model'
        char_path = path_prefix + ".char_vocab"
        label_path = path_prefix + ".label_vocab"
        POS_path = path_prefix + ".POS_vocab"
        NER_path = path_prefix + ".NER_vocab"
        has_pre_trained_model = False
        POS_vocab = None
        NER_vocab = None

    #if os.path.exists(best_path):
        if False == True:
            #has_pre_trained_model = True
            label_vocab = Vocab(label_path, fileformat='txt2')
            char_vocab = Vocab(char_path, fileformat='txt2')
            if FLAGS.with_POS: POS_vocab = Vocab(POS_path, fileformat='txt2')
            if FLAGS.with_NER: NER_vocab = Vocab(NER_path, fileformat='txt2')
        else:
            print('Collect words, chars and labels ...')
            (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path, with_POS=FLAGS.with_POS, with_NER=FLAGS.with_NER)
            print('Number of words: {}'.format(len(all_words)))
            print('Number of labels: {}'.format(len(all_labels)))
            label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2)
            label_vocab.dump_to_txt2(label_path)

            print('Number of chars: {}'.format(len(all_chars)))
            char_vocab = Vocab(fileformat='voc', voc=all_chars,dim=FLAGS.char_emb_dim)
            char_vocab.dump_to_txt2(char_path)

            if FLAGS.with_POS:
                print('Number of POSs: {}'.format(len(all_POSs)))
                POS_vocab = Vocab(fileformat='voc', voc=all_POSs,dim=FLAGS.POS_dim)
                POS_vocab.dump_to_txt2(POS_path)
            if FLAGS.with_NER:
                print('Number of NERs: {}'.format(len(all_NERs)))
                NER_vocab = Vocab(fileformat='voc', voc=all_NERs,dim=FLAGS.NER_dim)
                NER_vocab.dump_to_txt2(NER_path)


        print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
        print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
        num_classes = label_vocab.size()

        print('Build SentenceMatchDataStream ... ')

        is_list_wise = False
        if FLAGS.prediction_mode == "list_wise":
            is_list_wise = True

        if FLAGS.use_model_neg_sample == False:
            trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                      POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                      batch_size=FLAGS.batch_size, isShuffle=True, isLoop=True, isSort=True,
                                                      max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length,
                                                      is_as=FLAGS.is_answer_selection, is_word_overlap=FLAGS.word_overlap,
                                                     is_lemma_overlap= FLAGS.lemma_overlap, is_list_wise=is_list_wise,
                                                      min_answer_size=FLAGS.min_answer_size, max_answer_size = FLAGS.max_answer_size,
                                                      use_box = FLAGS.use_box,
                                                      sample_neg_from_question = FLAGS.nsfq,
                                                      equal_box_per_batch = FLAGS.equal_box_per_batch,
                                                      ) # box is just used for training


            train_testDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length,
                                                  is_as=FLAGS.is_answer_selection, is_word_overlap=FLAGS.word_overlap,
                                                 is_lemma_overlap= FLAGS.lemma_overlap)

        testDataStream = SentenceMatchDataStream(test_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length,
                                                  is_as=FLAGS.is_answer_selection, is_word_overlap=FLAGS.word_overlap,
                                                 is_lemma_overlap= FLAGS.lemma_overlap)


        devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab,
                                                  batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True,
                                                  max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length,
                                                  is_as=FLAGS.is_answer_selection, is_word_overlap=FLAGS.word_overlap,
                                                 is_lemma_overlap= FLAGS.lemma_overlap)


        if FLAGS.use_model_neg_sample == True:
            # init_scale = 0.01
            # with tf.Graph().as_default():
            #     initializer = tf.random_uniform_initializer(-init_scale, init_scale)

            trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                      POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                      label_vocab=label_vocab,
                                                      batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True,
                                                      isSort=True, # isShuffle=True means istrain=True but here we dont want for train
                                                      max_char_per_word=FLAGS.max_char_per_word,
                                                      max_sent_length=FLAGS.max_sent_length,
                                                      is_as=FLAGS.is_answer_selection,
                                                      is_word_overlap=FLAGS.word_overlap,
                                                      is_lemma_overlap=FLAGS.lemma_overlap,
                                                      is_list_wise=is_list_wise,
                                                      min_answer_size=FLAGS.min_answer_size,
                                                      max_answer_size=FLAGS.max_answer_size,
                                                      add_neg_sample_count = True, neg_sample_count=FLAGS.neg_sample_count)

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                    train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, with_char=not FLAGS.wo_char,
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
                                                  is_aggregation_lstm=FLAGS.is_aggregation_lstm
                                                  , max_window_size=FLAGS.max_window_size
                                                  , prediction_mode=FLAGS.prediction_mode,
                                                  context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
                                                  is_aggregation_siamese=FLAGS.is_aggregation_siamese
                                                  , unstack_cnn=FLAGS.unstack_cnn,
                                                  with_context_self_attention=FLAGS.with_context_self_attention,
                                                  mean_max=FLAGS.mean_max, clip_attention=FLAGS.clip_attention
                                                  , with_tanh=FLAGS.tanh)
            #
                    vars_ = {}
                    for var in tf.global_variables():
                        if "word_embedding" in var.name: continue
                        if not var.name.startswith("Model"): continue
                        vars_[var.name.split(":")[0]] = var
                    saver = tf.train.Saver(vars_)

                    sess = tf.Session()
                    sess.run(tf.global_variables_initializer())
                    step = 0
                    saver.restore(sess, best_path)
                    my_map, my_mrr, output_sentences, output_attention_weights = evaluate(trainDataStream, train_graph, sess,
                                                                                          char_vocab=char_vocab,
                                                                                          POS_vocab=POS_vocab,
                                                                                          NER_vocab=NER_vocab,
                                                                                          label_vocab=label_vocab,
                                                                                          flag_valid=False
                                                                                          , word_vocab=word_vocab,
                                                                                          first_on_best_model=True)

                    # my_map, my_mrr, output_sent, output_attention_weights = evaluate(trainDataStream, train_graph, sess,
                    #                                                                       char_vocab=char_vocab,
                    #                                                                       POS_vocab=POS_vocab,
                    #                                                                       NER_vocab=NER_vocab,
                    #                                                                       label_vocab=label_vocab,
                    #                                                                       flag_valid=True
                    #                                                                       ,word_vocab=word_vocab,
                    #                                                                       )


                    print ("train map on pretrain:", my_map)

                    # output_neg_file = open ("neg_file"+'run_id', 'wt')
                    #
                    # for zj in output_sent:
                    #     if sys.version_info[0] < 3:
                    #         output_neg_file.write(zj.encode('utf-8'))
                    #     else:
                    #         output_neg_file.write(zj)
                    #
                    # output_neg_file.close()


                    trainDataStream = SentenceMatchDataStream(output_sentences, word_vocab=word_vocab, char_vocab=char_vocab,
                                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                              label_vocab=label_vocab,
                                                              batch_size=FLAGS.batch_size, isShuffle=True, isLoop=True,
                                                              isSort=True, # shuf=true. now we want to train
                                                              max_char_per_word=FLAGS.max_char_per_word,
                                                              max_sent_length=FLAGS.max_sent_length,
                                                              is_as=FLAGS.is_answer_selection,
                                                              is_word_overlap=FLAGS.word_overlap,
                                                              is_lemma_overlap=FLAGS.lemma_overlap,
                                                              is_list_wise=is_list_wise,
                                                              min_answer_size=FLAGS.min_answer_size,
                                                              max_answer_size=FLAGS.max_answer_size,
                                                              add_neg_sample_count=False,
                                                              neg_sample_count=FLAGS.neg_sample_count,
                                                              use_top_negs = True,
                                                              train_from_path = False)
            #     accuracy, mrr = evaluate(testDataStream, valid_graph, sess,char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab
            #                         , mode='trec')

        print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
        print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
        print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
        print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
        print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
        print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

        sys.stdout.flush()
        if FLAGS.wo_char: char_vocab = None
        output_res_index = 1

        best_test_acc = 0
        while Generate_random_initialization(output_res_index) == True:
            st_cuda = ''
            if FLAGS.is_server == True:
                st_cuda = str(os.environ['CUDA_VISIBLE_DEVICES']) + '.'
            if 'trec' in test_path:
                ssst = 'tre' + FLAGS.run_id
            else:
                ssst = 'wik' + FLAGS.run_id
            ssst += str(FLAGS.start_batch)
            output_res_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index), 'wt')
            output_sentence_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index) + "S", 'wt')
            output_train_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index) + "T", 'wt')
            if FLAGS.store_att == True:
                output_attention_file = open('../result/' + ssst + '.'+ st_cuda + "A", 'wt')
            output_sentences = []
            output_res_index += 1
            output_res_file.write(str(FLAGS) + '\n\n')
            stt = str (FLAGS)
            best_dev_acc = 0.0
            init_scale = 0.01
            with tf.Graph().as_default():
                #initializer = tf.random_uniform_initializer(-init_scale, init_scale)
                initializer = tf.contrib.layers.xavier_initializer()
        #         with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                          dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                                                          lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim,
                                                          aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=True, MP_dim=FLAGS.MP_dim,
                                                          context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                          fix_word_vec=FLAGS.fix_word_vec, with_filter_layer=FLAGS.with_filter_layer, with_input_highway=FLAGS.with_highway,
                                                          word_level_MP_dim=FLAGS.word_level_MP_dim,
                                                          with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                          highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                          lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                          with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                                                          with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match),
                                                          with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
                                                          with_bilinear_att=(FLAGS.attention_type)
                                                          , type1=FLAGS.type1, type2 = FLAGS.type2, type3=FLAGS.type3,
                                                          with_aggregation_attention=not FLAGS.wo_agg_self_att,
                                                          is_answer_selection= FLAGS.is_answer_selection,
                                                          is_shared_attention=FLAGS.is_shared_attention,
                                                          modify_loss=FLAGS.modify_loss, is_aggregation_lstm=FLAGS.is_aggregation_lstm
                                                          , max_window_size=FLAGS.max_window_size
                                                          , prediction_mode=FLAGS.prediction_mode,
                                                          context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
                                                          is_aggregation_siamese=FLAGS.is_aggregation_siamese
                                                          , unstack_cnn=FLAGS.unstack_cnn,with_context_self_attention=FLAGS.with_context_self_attention,
                                                          mean_max=FLAGS.mean_max, clip_attention=FLAGS.clip_attention
                                                          ,with_tanh=FLAGS.tanh, new_list_wise=FLAGS.new_list_wise,
                                                          max_answer_size=FLAGS.max_answer_size, q_count=FLAGS.question_count_per_batch,
                                                          sampling=FLAGS.sampling, sampling_type=FLAGS.sampling_type,
                                                          sample_percent = FLAGS.sample_percent, top_treshold=FLAGS.top_treshold,
                                                          pos_avg=FLAGS.pos_avg)
                    tf.summary.scalar("Training Loss", train_graph.get_loss()) # Add a scalar summary for the snapshot loss.

        #         with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab,
                                                          dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                                                          lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim,
                                                          aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim,
                                                          context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                          fix_word_vec=FLAGS.fix_word_vec, with_filter_layer=FLAGS.with_filter_layer, with_input_highway=FLAGS.with_highway,
                                                          word_level_MP_dim=FLAGS.word_level_MP_dim,
                                                          with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                          highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                          lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                          with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                                                          with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match),
                                                          with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
                                                          with_bilinear_att=(FLAGS.attention_type)
                                                          , type1=FLAGS.type1, type2 = FLAGS.type2, type3=FLAGS.type3,
                                                          with_aggregation_attention=not FLAGS.wo_agg_self_att,
                                                          is_answer_selection= FLAGS.is_answer_selection,
                                                          is_shared_attention=FLAGS.is_shared_attention,
                                                          modify_loss=FLAGS.modify_loss, is_aggregation_lstm=FLAGS.is_aggregation_lstm,
                                                          max_window_size=FLAGS.max_window_size
                                                          , prediction_mode=FLAGS.prediction_mode,
                                                          context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
                                                          is_aggregation_siamese=FLAGS.is_aggregation_siamese
                                                          , unstack_cnn=FLAGS.unstack_cnn,with_context_self_attention=FLAGS.with_context_self_attention,
                                                          mean_max=FLAGS.mean_max, clip_attention=FLAGS.clip_attention
                                                          ,with_tanh=FLAGS.tanh, new_list_wise=FLAGS.new_list_wise,
                                                          q_count=1, pos_avg = FLAGS.pos_avg)


                initializer = tf.global_variables_initializer()
                vars_ = {}
                #for var in tf.all_variables():
                for var in tf.global_variables():
                    if "word_embedding" in var.name: continue
        #             if not var.name.startswith("Model"): continue
                    vars_[var.name.split(":")[0]] = var
                saver = tf.train.Saver(vars_)

                with tf.Session() as sess:
                    sess.run(initializer)
                    # if FLAGS.use_model_neg_sample == True:
                    #     print("Restoring model from " + best_path)
                    #     saver.restore(sess, best_path)
                    #     print("DONE!")


                    print('Start the training loop.')
                    train_size = trainDataStream.get_num_batch()
                    max_steps = (train_size * FLAGS.max_epochs) // FLAGS.question_count_per_batch
                    epoch_size = max_steps // (FLAGS.max_epochs*2) + 1
                    #max_steps += (train_size * FLAGS.max_epochs) % FLAGS.question_count_per_batch
                    #max_steps = 2
                    total_loss = 0.0
                    start_time = time.time()

                    max_valid = 0
                    flag_next_epoch = False
                    for step in range(max_steps):

                        # read data
                        _truth = []
                        _question_lengths = []
                        _passage_lengths = []
                        _in_question_words = []
                        _in_passage_words = []
                        _overlap = []
                        _question_count = []
                        _answer_count = []
                        _hinge_truth = []
                        _real_answer_count_mask = []

                        _mask = []
                        _mask_topk = []
                        for i in range (FLAGS.question_count_per_batch):
                            # if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                            #     break
                            if step != 0 and trainDataStream.cur_pointer < FLAGS.question_count_per_batch:
                                flag_next_epoch = True
                            cur_batch, batch_index = trainDataStream.nextBatch()
                            (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                                             char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch,
                                             sent1_char_length_batch, sent2_char_length_batch,
                                             POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch) = cur_batch

                            if FLAGS.prediction_mode == 'list_mle':
                                (label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch,
                                 word_idx_2_batch,
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch,
                                 sent2_length_batch,
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch) = sort_mle(label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                                             char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch,
                                             sent1_char_length_batch, sent2_char_length_batch,
                                             POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch)
                            _truth.append(label_id_batch)
                            _question_lengths.append(sent1_length_batch)
                            _passage_lengths.append(sent2_length_batch)
                            _in_question_words.append(word_idx_1_batch)
                            _in_passage_words.append(word_idx_2_batch)
                            _overlap.append(overlap_batch)
                            _question_count.append(trainDataStream.question_count(batch_index))
                            _answer_count.append(trainDataStream.answer_count(batch_index))
                            #_hinge_truth.append(make_hinge_truth(label_id_batch, trainDataStream.question_count(batch_index),
                            #                                                            trainDataStream.answer_count(
                            #                                                                batch_index)))
                            _real_answer_count_mask.append(trainDataStream.real_answer_count(batch_index))
                            if FLAGS.prediction_mode == 'list_mle':
                                _mask.append(get_mle_mask(len(label_id_batch)))
                                _mask_topk.append(get_mle_mask_topk (FLAGS.topk , len (label_id_batch)))

                            #print (label_id_batch)
                            #print (_mask[i])
                            #print (_mask_topk[i])

                        feed_dict = {
                                 train_graph.get_truth(): tuple(_truth),
                                 train_graph.get_question_lengths(): tuple(_question_lengths),
                                 train_graph.get_passage_lengths(): tuple(_passage_lengths),
                                 train_graph.get_in_question_words(): tuple(_in_question_words),
                                 train_graph.get_in_passage_words(): tuple(_in_passage_words),
                                 train_graph.get_overlap(): tuple(_overlap)
                                #train_graph.get_mask ():tuple(_mask),
                                #train_graph.get_mask_topk ():(_mask_topk)
        #                          train_graph.get_question_char_lengths(): sent1_char_length_batch,
        #                          train_graph.get_passage_char_lengths(): sent2_char_length_batch,
        #                          train_graph.get_in_question_chars(): char_matrix_idx_1_batch,
        #                          train_graph.get_in_passage_chars(): char_matrix_idx_2_batch
                                 }
                        if FLAGS.prediction_mode == 'list_mle':
                            feed_dict[train_graph.get_mask()] = tuple(_mask)
                            feed_dict[train_graph.get_mask_topk()] = tuple(_mask_topk)


                        # if char_vocab is not None and FLAGS.wo_char == False:
                    #     feed_dict[train_graph.get_question_char_lengths()] = sent1_char_length_batch
                    #     feed_dict[train_graph.get_passage_char_lengths()] = sent2_char_length_batch
                    #     feed_dict[train_graph.get_in_question_chars()] = char_matrix_idx_1_batch
                    #     feed_dict[train_graph.get_in_passage_chars()] = char_matrix_idx_2_batch
                    #
                    # if POS_vocab is not None:
                    #     feed_dict[train_graph.get_in_question_poss()] = POS_idx_1_batch
                    #     feed_dict[train_graph.get_in_passage_poss()] = POS_idx_2_batch
                    #
                    # if NER_vocab is not None:
                    #     feed_dict[train_graph.get_in_question_ners()] = NER_idx_1_batch
                    #     feed_dict[train_graph.get_in_passage_ners()] = NER_idx_2_batch

                        if FLAGS.is_answer_selection == True:
                            feed_dict[train_graph.get_question_count()] = tuple(_question_count)
                            feed_dict[train_graph.get_answer_count()] = tuple (_answer_count)
                            #feed_dict[train_graph.get_hinge_truth()] = tuple (_hinge_truth)
                            feed_dict[train_graph.get_real_answer_count_mask()] = tuple (_real_answer_count_mask)

                        _, loss_value = sess.run([train_graph.get_train_op(), train_graph.get_loss()], feed_dict=feed_dict)
                        total_loss += loss_value
                        #if FLAGS.is_answer_selection == True and FLAGS.is_server == False:
                        #    print ("q: {} a: {} loss_value: {}".format(trainDataStream.question_count(batch_index)
                        #                               ,trainDataStream.answer_count(batch_index), loss_value))

                        if step % 50==0:
                            print('{} '.format(step), end="")
                            sys.stdout.flush()

                        # Save a checkpoint and evaluate the model periodically.
                        if (step+1) % epoch_size == 0 or (step + 1) == max_steps:
                            flag_next_epoch = False
                            #print(total_loss)
                            # Print status to stdout.
                            duration = time.time() - start_time
                            start_time = time.time()
                            output_res_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                            total_loss = 0.0

                            #Evaluate against the validation set.
                            output_res_file.write('valid- ')
                            my_map, my_mrr = evaluate(devDataStream, valid_graph, sess,char_vocab=char_vocab,
                                                POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab)
                            output_res_file.write("map: '{}', mrr: '{}'\n".format(my_map, my_mrr))
                            #print ("dev map: {}".format(my_map))
                            #print("Current accuracy is %.2f" % accuracy)

                            #accuracy = my_map
                            #if accuracy>best_accuracy:
                            #    best_accuracy = accuracy
                            #    saver.save(sess, best_path)

                            # Evaluate against the test set.
                            flag_valid = False
                            if my_map > max_valid:
                                max_valid = my_map
                                flag_valid = True

                            output_res_file.write ('test- ')
                            if flag_valid == True:
                                my_map, my_mrr, output_sentences, output_attention_weights = evaluate(testDataStream, valid_graph, sess, char_vocab=char_vocab,
                                     POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, flag_valid=flag_valid
                                                                            ,word_vocab=word_vocab)
                                if my_map > best_test_acc and FLAGS.store_best == True:
                                    best_test_acc = my_map
                                    saver.save(sess, best_path)

                                #tr_map, tr_mrr, train_sentences, _ = evaluate(train_testDataStream, valid_graph, sess, char_vocab=char_vocab,
                                     #POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, flag_valid=True
                                     #                                         ,word_vocab=word_vocab)
                            else:
                                my_map,my_mrr = evaluate(testDataStream, valid_graph, sess, char_vocab=char_vocab,
                                     POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, flag_valid=flag_valid)

                            output_res_file.write("map: '{}', mrr: '{}\n\n".format(my_map, my_mrr))


                            flag_valid = False
                            #if FLAGS.is_server == False:
                            print ("test map: {}".format(my_map))

                            #Evaluate against the train set only for final epoch.
                            if (step + 1) == max_steps:
                                output_res_file.write ('train- ')
                                # my_map, my_mrr = evaluate(train_testDataStream, valid_graph, sess, char_vocab=char_vocab,
                                #     POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab)

                                # my_map,my_mrr = evaluate(train_testDataStream, valid_graph, sess, char_vocab=char_vocab,
                                #      POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, flag_valid=False)


                                output_res_file.write("map: '{}', mrr: '{}'\n".format(0.0, 0.0))#(tr_map, tr_mrr))
                                print ("map: '{}', mrr: '{}'\n".format(my_map, my_mrr))




            # print("Best accuracy on dev set is %.2f" % best_accuracy)
            # # decoding
            # print('Decoding on the test set:')
            # init_scale = 0.01
            # with tf.Graph().as_default():
            #     initializer = tf.random_uniform_initializer(-init_scale, init_scale)
            #     with tf.variable_scope("Model", reuse=False, initializer=initializer):
            #         valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab,
            #              dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
            #              lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim,
            #              aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim,
            #              context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num,
            #              fix_word_vec=FLAGS.fix_word_vec,with_filter_layer=FLAGS.with_filter_layer, with_highway=FLAGS.with_highway,
            #              word_level_MP_dim=FLAGS.word_level_MP_dim,
            #              with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
            #              highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition,
            #              lex_decompsition_dim=FLAGS.lex_decompsition_dim,
            #              with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
            #              with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match),
            #              with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
            #                                               with_bilinear_att=(not FLAGS.wo_bilinear_att)
            #                                               , type1=FLAGS.type1, type2 = FLAGS.type2, type3=FLAGS.type3,
            #                                               with_aggregation_attention=not FLAGS.wo_agg_self_att,
            #                                               is_answer_selection= FLAGS.is_answer_selection,
            #                                               is_shared_attention=FLAGS.is_shared_attention,
            #                                               modify_loss=FLAGS.modify_loss,is_aggregation_lstm=FLAGS.is_aggregation_lstm,
            #                                               max_window_size=FLAGS.max_window_size,
            #                                               prediction_mode=FLAGS.prediction_mode,
            #                                               context_lstm_dropout=not FLAGS.wo_lstm_drop_out,
            #                                              is_aggregation_siamese=FLAGS.is_aggregation_siamese)
            #
            #     vars_ = {}
            #     for var in tf.global_variables():
            #         if "word_embedding" in var.name: continue
            #         if not var.name.startswith("Model"): continue
            #         vars_[var.name.split(":")[0]] = var
            #     saver = tf.train.Saver(vars_)
            #
            #     sess = tf.Session()
            #     sess.run(tf.global_variables_initializer())
            #     step = 0
            #     saver.restore(sess, best_path)
            #
            #     accuracy, mrr = evaluate(testDataStream, valid_graph, sess,char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab
            #                         , mode='trec')
            #     output_res_file.write("map for test set is %.2f\n" % accuracy)


            for zj in output_sentences:
                if sys.version_info[0] < 3:
                    output_sentence_file.write(zj.encode('utf-8'))
                else:
                    output_sentence_file.write(zj)

            #for zj in train_sentences:
            #    if sys.version_info[0] < 3:
            #        output_train_file.write(zj.encode('utf-8'))
            #    else:
            #        output_train_file.write(zj)

            output_train_file.close()
            output_sentence_file.close()
            output_res_file.close()

            if FLAGS.store_att == True:
                for zj in output_attention_weights:
                    output_attention_file.write(zj)
                FLAGS.store_att = False

        FLAGS.start_batch += FLAGS.step_batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_trec',default=False, help='is trec or wiki?')
    FLAGS, unparsed = parser.parse_known_args()
    is_trec = FLAGS.is_trec
    if is_trec == 'True' or is_trec == True:
        is_trec = True
    else:
        is_trec = False
    if is_trec == True:
        qa_path = 'trecqa/'
    else:
        qa_path = 'wikiqa/WikiQACorpus/WikiQA-'
    parser.add_argument('--word_vec_path', type=str, default='../data/glove/glove.6B.50d.txt', help='Path the to pre-trained word vector model.')
    #parser.add_argument('--word_vec_path', type=str, default='../data/glove/glove.840B.300d.txt', help='Path the to pre-trained word vector model.')
    parser.add_argument('--is_server',default=False, type= bool, help='do we have cuda visible devices?')
    parser.add_argument('--is_random_init',default=True, help='loop: ranom initalizaion of parameters -> run ?')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum epochs for training.')
    parser.add_argument('--attention_type', default='dot_product', help='[bilinear, linear, linear_p_bias, dot_product]')


    parser.add_argument('--use_model_neg_sample',default=False, type= bool, help='do we have cuda visible devices?')
    parser.add_argument('--neg_sample_count',default=100, type= int, help='do we have cuda visible devices?')
    parser.add_argument('--store_best',default=False, type = bool, help='do we have cuda visible devices?')



    parser.add_argument('--use_box',default=True, type= bool, help='do we have cuda visible devices?')
    parser.add_argument('--nsfq',default=True, help='negative sample from question')
    parser.add_argument('--new_list_wise', default=True, help='do we have cuda visible devices?')

    #FLAGS, unparsed = parser.parse_known_args()

    parser.add_argument('--top_treshold', type=int, default=-1, help='Maximum epochs for training.')


    parser.add_argument('--start_batch', type=int, default=0, help='Maximum epochs for training.')
    parser.add_argument('--end_batch', type=int, default=0, help='Maximum epochs for training.')
    parser.add_argument('--step_batch', type=int, default=1, help='Maximum epochs for training.')



    parser.add_argument('--sampling',default=True, help='for loss back prop')
    parser.add_argument('--sampling_type',default='attentive', help='for loss back prop')
    parser.add_argument('--sample_percent',default=0.8, help='for loss back prop')





    parser.add_argument('--equal_box_per_batch',default=True, help='do we have cuda visible devices?')


    parser.add_argument('--store_att',default=False, type= bool, help='do we have cuda visible devices?')

    parser.add_argument('--pos_avg',default=True, type= bool, help='do we have cuda visible devices?')



    #bs = 100 #135 #110
    #if is_trec == False:
    #    bs = 40

    parser.add_argument('--question_count_per_batch', type=int, default= 4, help='Number of instances in each batch.')


    parser.add_argument('--min_answer_size', type=int, default= 0, help='Number of instances in each batch.')
    parser.add_argument('--max_answer_size', type=int, default= 150, help='Number of instances in each batch.')

    #question_per_batch = 1

    FLAGS, unparsed = parser.parse_known_args()

    bs = FLAGS.max_answer_size

    parser.add_argument('--batch_size', type=int, default=bs, help='Number of instances in each batch.')
    parser.add_argument('--is_answer_selection',default=True, type =bool, help='is answer selection or other sentence matching tasks?')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--prediction_mode', default='list_wise', help = 'point_wise, list_wise, hinge_wise .'
                                                                          'point wise is only used for non answer selection tasks')

    parser.add_argument('--train_path', type=str,default = '../data/' +qa_path +'train.txt', help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default = '../data/' + qa_path +'dev.txt', help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, default = '../data/'+qa_path+'test.txt',help='Path to the test set.')
    parser.add_argument('--model_dir', type=str,default = '../models',help='Directory to save model files.')

    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.000001, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.05, help='Dropout ratio.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=50, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--highway_layer_num', type=int, default=0, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', required=False, help='Suffix of the model name.')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--wo_char', default=True, help='Without character-composed embeddings.', action='store_true')
    parser.add_argument('--type1', default= 'w_sub_mul', help='similrty function 1', action='store_true')
    parser.add_argument('--type2', default= None , help='similrty function 2', action='store_true')
    parser.add_argument('--type3', default= None , help='similrty function 3', action='store_true')
    parser.add_argument('--wo_lstm_drop_out', default=  True , help='with out context lstm drop out', action='store_true')
    parser.add_argument('--wo_agg_self_att', default= True , help='with out aggregation lstm self attention', action='store_true')
    parser.add_argument('--is_shared_attention', default= False , help='are matching attention values shared or not', action='store_true')
    parser.add_argument('--modify_loss', type=float, default=0, help='a parameter used for loss.')
    parser.add_argument('--is_aggregation_lstm', default=True, help = 'is aggregation lstm or aggregation cnn' )
    parser.add_argument('--max_window_size', type=int, default=2, help = '[1..max_window_size] convolution')
    parser.add_argument('--is_aggregation_siamese', default=True, help = 'are aggregation wieghts on both sides shared or not' )
    parser.add_argument('--unstack_cnn', default=False, help = 'are aggregation wieghts on both sides shared or not' )
    parser.add_argument('--with_context_self_attention', default=False, help = 'are aggregation wieghts on both sides shared or not' )


    parser.add_argument('--MP_dim', type=int, default=50, help='Number of perspectives for matching vectors.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--with_highway', default=True, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')


    parser.add_argument('--word_overlap', default=False, help = 'are aggregation wieghts on both sides shared or not' )
    parser.add_argument('--lemma_overlap', default=False, help = 'are aggregation wieghts on both sides shared or not' )    #if (len (st) >=2 and st [1] == '.') : continue


    parser.add_argument('--mean_max', default=False, help = 'are aggregation wieghts on both sides shared or not' )
    parser.add_argument('--clip_attention', default=False, help = 'are aggregation wieghts on both sides shared or not' )

    parser.add_argument('--tanh', default=False , help = 'just ignore. this is a shit')



    #these parameters arent used anymore:
    parser.add_argument('--with_filter_layer', default=False, help='Utilize filter layer.', action='store_true')
    parser.add_argument('--word_level_MP_dim', type=int, default=-1, help='Number of perspectives for word-level matching.')
    parser.add_argument('--with_lex_decomposition', default=False, help='Utilize lexical decomposition features.',
                        action='store_true')
    parser.add_argument('--lex_decompsition_dim', type=int, default=-1,
                        help='Number of dimension for lexical decomposition features.')
    parser.add_argument('--with_POS', default=False, help='Utilize POS information.', action='store_true')
    parser.add_argument('--with_NER', default=False, help='Utilize NER information.', action='store_true')
    parser.add_argument('--POS_dim', type=int, default=20, help='Number of dimension for POS embeddings.')
    parser.add_argument('--NER_dim', type=int, default=20, help='Number of dimension for NER embeddings.')
    parser.add_argument('--wo_left_match', default=False, help='Without left to right matching.', action='store_true')
    parser.add_argument('--wo_right_match', default=False, help='Without right to left matching', action='store_true')
    parser.add_argument('--wo_full_match', default=True, help='Without full matching.', action='store_true')
    parser.add_argument('--wo_maxpool_match', default=True, help='Without maxpooling matching', action='store_true')
    parser.add_argument('--wo_attentive_match', default=True, help='Without attentive matching', action='store_true')
    parser.add_argument('--wo_max_attentive_match', default=True, help='Without max attentive matching.',
                        action='store_true')
    parser.add_argument('--fix_word_vec', default=True, help='Fix pre-trained word embeddings during training.', action='store_true')



    parser.add_argument('--run_id', default='10' , help = 'run_id')

    #     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.nsfq == 'True' or  FLAGS.nsfq == True:
        FLAGS.nsfq = True
    else:
        FLAGS.nsfq = False
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

