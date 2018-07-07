import tensorflow as tf
#import my_rnn
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
import match_utils

eps = 1e-8


def get_place_holder (q_count, type, shape):
    ans = ()
    for _ in range(q_count):
        ans += (tf.placeholder(type, shape=shape),)
    return ans


class SentenceMatchModelGraph(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None,
                 dropout_rate=0.5, learning_rate=0.001, optimize_type='adam', lambda_l2=1e-5,
                 with_word=True, with_char=True, with_POS=True, with_NER=True,
                 char_lstm_dim=20, context_lstm_dim=100, aggregation_lstm_dim=200, is_training=True, filter_layer_threshold=0.2,
                 MP_dim=50, context_layer_num=1, aggregation_layer_num=1, fix_word_vec=False, with_filter_layer=True, with_input_highway=False,
                 with_lex_features=False, lex_dim=100, word_level_MP_dim=-1, sep_endpoint=False, end_model_combine=False, with_match_highway=False,
                 with_aggregation_highway=False, highway_layer_num=1, with_lex_decomposition=False, lex_decompsition_dim=-1,
                 with_left_match=True, with_right_match=True,
                 with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                 with_bilinear_att = 's', type1 = None, type2 = None, type3 = None, with_aggregation_attention = True,
                 is_answer_selection = True, is_shared_attention = True, modify_loss = 0, is_aggregation_lstm = True, max_window_size=3
                 , prediction_mode = 'list_wise', context_lstm_dropout = True, is_aggregation_siamese = True, unstack_cnn = True,with_context_self_attention=False,
                 clip_attention = True, mean_max = True, with_tanh = True , new_list_wise=True, max_answer_size = 15,
                 q_count=2, pos_avg = True, sampling = False, sampling_type = 'attentive', sample_percent = 0.8,
                 top_treshold = -1):

        # ======word representation layer======

        self.question_lengths = get_place_holder (q_count, tf.int32, [None])#tf.placeholder(tf.int32, [None])]
        self.passage_lengths = get_place_holder (q_count, tf.int32, [None])#tf.placeholder(tf.int32, [None])]
        if is_answer_selection == True:
            self.truth = get_place_holder (q_count, tf.float32, [None])#q_count*[tf.placeholder(tf.float32, [None])] # [batch_size]
            self.hinge_truth = get_place_holder (q_count, tf.float32, [None, None, None])#q_count*[tf.placeholder(tf.float32, [None, None, None])] #[Q,A,A]
            self.real_answer_count_mask = get_place_holder (q_count, tf.float32, [None, None])#q_count*[tf.placeholder(tf.float32, [None, None])]
        else:
            self.truth = get_place_holder (q_count, tf.int32, [None])#tf.placeholder(tf.int32, [None]) # [batch_size]

        self.question_count = get_place_holder (q_count, tf.int32, None)#q_count*[tf.placeholder(tf.int32, None)]
        self.answer_count = get_place_holder (q_count, tf.int32, None)#q_count*[tf.placeholder(tf.int32, None)]

        self.overlap = get_place_holder (q_count, tf.float32, [None, None, None])#q_count*[tf.placeholder(tf.float32, [None, None, None])] #[bs, N, M]

        #if with_word and word_vocab is not None:
        self.in_question_words = get_place_holder (q_count, tf.int32, [None, None])#q_count*[tf.placeholder(tf.int32, [None, None])] # [batch_size, question_len]
        self.in_passage_words = get_place_holder (q_count, tf.int32, [None, None])#q_count*[tf.placeholder(tf.int32, [None, None])] # [batch_size, passage_len]
#             self.word_embedding = tf.get_variable("word_embedding", shape=[word_vocab.size()+1, word_vocab.word_dim], initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
        word_vec_trainable = True
        cur_device = '/gpu:0'
        if fix_word_vec:
            word_vec_trainable = False
            cur_device = '/cpu:0'
        with tf.device(cur_device):
            self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                              initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

        match_representation_list = []
        loss_list = []
        score_list = []
        prob_list = []
        pos_list = []
        with tf.variable_scope ('salamzendegi'):
            for i in range (q_count):
                input_dim = 0
                in_question_repres = []
                in_passage_repres = []
                in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words[i]) # [batch_size, question_len, word_dim]
                in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words[i]) # [batch_size, passage_len, word_dim]
                in_question_repres.append(in_question_word_repres)
                in_passage_repres.append(in_passage_word_repres)

                input_shape = tf.shape(self.in_question_words[i])
                batch_size = input_shape[0]
                question_len = input_shape[1]
                input_shape = tf.shape(self.in_passage_words[i])
                passage_len = input_shape[1]
                input_dim += word_vocab.word_dim

    #         if with_POS and POS_vocab is not None:
    #             self.in_question_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
    #             self.in_passage_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
    # #             self.POS_embedding = tf.get_variable("POS_embedding", shape=[POS_vocab.size()+1, POS_vocab.word_dim], initializer=tf.constant(POS_vocab.word_vecs), dtype=tf.float32)
    #             self.POS_embedding = tf.get_variable("POS_embedding", initializer=tf.constant(POS_vocab.word_vecs), dtype=tf.float32)
    #
    #             in_question_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_question_POSs) # [batch_size, question_len, POS_dim]
    #             in_passage_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_passage_POSs) # [batch_size, passage_len, POS_dim]
    #             in_question_repres.append(in_question_POS_repres)
    #             in_passage_repres.append(in_passage_POS_repres)
    #
    #             input_shape = tf.shape(self.in_question_POSs)
    #             batch_size = input_shape[0]
    #             question_len = input_shape[1]
    #             input_shape = tf.shape(self.in_passage_POSs)
    #             passage_len = input_shape[1]
    #             input_dim += POS_vocab.word_dim
    #
    #         if with_NER and NER_vocab is not None:
    #             self.in_question_NERs = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
    #             self.in_passage_NERs = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
    # #             self.NER_embedding = tf.get_variable("NER_embedding", shape=[NER_vocab.size()+1, NER_vocab.word_dim], initializer=tf.constant(NER_vocab.word_vecs), dtype=tf.float32)
    #             self.NER_embedding = tf.get_variable("NER_embedding", initializer=tf.constant(NER_vocab.word_vecs), dtype=tf.float32)
    #
    #             in_question_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_question_NERs) # [batch_size, question_len, NER_dim]
    #             in_passage_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_passage_NERs) # [batch_size, passage_len, NER_dim]
    #             in_question_repres.append(in_question_NER_repres)
    #             in_passage_repres.append(in_passage_NER_repres)
    #
    #             input_shape = tf.shape(self.in_question_NERs)
    #             batch_size = input_shape[0]
    #             question_len = input_shape[1]
    #             input_shape = tf.shape(self.in_passage_NERs)
    #             passage_len = input_shape[1]
    #             input_dim += NER_vocab.word_dim
    #
    #         if with_char and char_vocab is not None:
    #             self.question_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
    #             self.passage_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
    #             self.in_question_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
    #             self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]
    #             input_shape = tf.shape(self.in_question_chars)
    #             batch_size = input_shape[0]
    #             question_len = input_shape[1]
    #             q_char_len = input_shape[2]
    #             input_shape = tf.shape(self.in_passage_chars)
    #             passage_len = input_shape[1]
    #             p_char_len = input_shape[2]
    #             char_dim = char_vocab.word_dim
    # #             self.char_embedding = tf.get_variable("char_embedding", shape=[char_vocab.size()+1, char_vocab.word_dim], initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
    #             self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
    #
    #             in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
    #             in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
    #             question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
    #             in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
    #             in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
    #             passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
    #             with tf.variable_scope('char_lstm'):
    #                 # lstm cell
    #                 #char_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(char_lstm_dim)
    #                 char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(char_lstm_dim)
    #                 # dropout
    #                 #if is_training: char_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - dropout_rate))
    #                 #char_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell])
    #                 if is_training: char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell,
    #                                                                                output_keep_prob=(1 - dropout_rate))
    #                 char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])
    #
    #                 # question_representation
    #                 question_char_outputs = dynamic_rnn(char_lstm_cell, in_question_char_repres,
    #                         sequence_length=question_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
    #                 question_char_outputs = question_char_outputs[:,-1,:]
    #                 question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, char_lstm_dim])
    #
    #                 tf.get_variable_scope().reuse_variables()
    #                 # passage representation
    #                 passage_char_outputs = dynamic_rnn(char_lstm_cell, in_passage_char_repres,
    #                         sequence_length=passage_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
    #                 passage_char_outputs = passage_char_outputs[:,-1,:]
    #                 passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, char_lstm_dim])
    #
    #             in_question_repres.append(question_char_outputs)
    #             in_passage_repres.append(passage_char_outputs)
    #
    #             input_dim += char_lstm_dim
                in_question_repres = tf.concat(in_question_repres, 2) # [batch_size, question_len, dim]
                in_passage_repres = tf.concat(in_passage_repres, 2) # [batch_size, passage_len, dim]

                if is_training:
                   in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
                   in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))
                else:
                   in_question_repres = tf.multiply(in_question_repres, (1 - dropout_rate))
                   in_passage_repres = tf.multiply(in_passage_repres, (1 - dropout_rate))

                mask = tf.sequence_mask(self.passage_lengths[i], passage_len, dtype=tf.float32) # [batch_size, passage_len]
                question_mask = tf.sequence_mask(self.question_lengths[i], question_len, dtype=tf.float32) # [batch_size, question_len]

            # ======Highway layer======
                if with_input_highway:
                    with tf.variable_scope("input_highway"):
                        output_size = context_lstm_dim
                        flag_highway = False
                        if context_layer_num == 2:
                            output_size = input_dim
                            flag_highway = True
                        in_question_repres = match_utils.highway_layer(in_question_repres, input_size=input_dim, scope='s',
                                                                       output_size=output_size,with_highway=flag_highway)
                        tf.get_variable_scope().reuse_variables()
                        in_passage_repres = match_utils.highway_layer(in_passage_repres, input_size=input_dim, scope='s',
                                                                       output_size=output_size, with_highway=flag_highway)
                    # if context_layer_num == 2:
                    #     with tf.variable_scope("input_highway2"):
                    #         context_layer_num = 1
                    #         in_question_repres = match_utils.highway_layer(in_question_repres, input_size=output_size, scope='s1',
                    #                                                        output_size=context_lstm_dim)
                    #         a1 = tf.get_variable("a1", [5, 3], dtype=tf.float32)
                    #
                    #         tf.get_variable_scope().reuse_variables()
                    #         in_passage_repres = match_utils.highway_layer(in_passage_repres, input_size=output_size, scope='s1',
                    #                                                           output_size=context_lstm_dim)
            # ========Bilateral Matching=====

                (match_representation, match_dim, self.attention_weights) = match_utils.bilateral_match_func2(in_question_repres, in_passage_repres,
                                self.question_lengths[i], self.passage_lengths[i], question_mask, mask, MP_dim, input_dim,
                                with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                                with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                                with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                                with_full_match, with_maxpool_match, with_attentive_match, with_max_attentive_match,
                                with_left_match, with_right_match, with_bilinear_att, type1, type2, type3, with_aggregation_attention
                                                                                      ,is_shared_attention, is_aggregation_lstm,
                                                                                      max_window_size, context_lstm_dropout,
                                                                                      is_aggregation_siamese,unstack_cnn, with_input_highway,with_context_self_attention,
                                                                                      self.overlap[i], mean_max, clip_attention)
                #match_representation_list.append(match_representation)
            #========Prediction Layer=========
                pred_highway = True
                if pred_highway == False:
                    w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
                    b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
                    logits = tf.matmul(match_representation, w_0) + b_0
                    logits = tf.tanh(logits)
                else:
                    logits=match_utils.highway_layer(in_val=tf.expand_dims(match_representation,0), input_size=match_dim,output_size=match_dim,with_highway=True
                                              ,scope='pred_highway') #[1, bs, d]
                    logits = tf.reduce_sum(logits, 0) #[bs,d]
                    match_dim *= 2
                if is_training:
                    logits = tf.nn.dropout(logits, (1 - dropout_rate))
                else:
                    logits = tf.multiply(logits, (1 - dropout_rate))

                sec_dim = 1
                if prediction_mode == 'point_wise':
                    sec_dim = 2

                w_1 = tf.get_variable("w_1", [match_dim/2, sec_dim],dtype=tf.float32)
                b_1 = tf.get_variable("b_1", [sec_dim],dtype=tf.float32)
                logits = tf.matmul(logits, w_1) + b_1
                #self.score = tf.reshape(logits, [-1])
                if prediction_mode != 'point_wise':
                    score_list.append(tf.reshape(logits, [-1]))
                    logits = tf.reshape(logits, shape=[self.question_count[i], self.answer_count[i]])
                    gold_matrix = tf.reshape(self.truth[i], shape=[self.question_count[i], self.answer_count[i]])
                    g1_matrix = tf.ceil(gold_matrix - eps)
                    if prediction_mode == 'list_wise':
                        # p: truth label
                        # q: prediction
                        # H(p,q) = sum(p(x)log(p(x))) - sum(p(x)log(q(x))
                        # loss = mean(H(p,q)) for p,q in batch
                        #self.prob = tf.reshape(logits, [-1]) #[bs]
                        prob_list.append(tf.reshape(logits, [-1]))
                        if new_list_wise == False:
                            logits = tf.multiply(logits, self.real_answer_count_mask[i])
                            logits = tf.nn.softmax(logits)  # [question_count, answer_count]
                            #logits = tf.multiply(logits, self.real_answer_count_mask)
                            #logits = tf.exp(logits)
                            # input_shape = tf.shape(logits)
                            # ans_count = input_shape[1]
                            #logits_sum = tf.reduce_sum(logits) + \
                            #             (float(max_answer_size) - tf.cast(self.answer_count[i], tf.float32))
                            #logits = logits / logits_sum
                            loss_list.append(tf.reduce_sum(
                                tf.multiply(gold_matrix, tf.log(gold_matrix+eps)) - tf.multiply(gold_matrix, tf.log(logits))
                                ))
                            #ModifyLoss:
                            #alpha1 = tf.reduce_mean(tf.reduce_sum(tf.multiply(g1_matrix, logits), axis=1))
                            #self.loss += modify_loss / alpha1
                        else:

                            # prob_distribution = tf.nn.softmax(logits)
                            # log_prpb_distribution = tf.log (prob_distribution)
                            # samples = tf.multifnomial(log_prpb_distribution, max_sample_size) #[1, max_sample_size]

                            logits = tf.reshape(logits, [-1])
                            g1_matrix = tf.reshape(g1_matrix, [-1])
                            input_shape = tf.shape(g1_matrix)[0]
                            input_shape = tf.cast(input_shape, tf.int32)
                            pos_mask = g1_matrix #[a]
                            neg_mask = 1 - g1_matrix #[a]
                            neg_count = tf.reduce_sum(neg_mask) #[1]
                            pos_count = tf.reduce_sum(pos_mask) #[1]

                            if top_treshold < 0:
                                pos_sample_size = pos_count
                                neg_sample_size = tf.minimum(float(sample_percent), neg_count)
                                neg_sample_size = tf.cast(neg_sample_size, tf.int32)
                                if sampling == True:
                                    if sampling_type == 'random':
                                        pos_prob = tf.divide (pos_mask, pos_count)
                                        neg_prob = tf.divide(neg_mask, neg_count)
                                    elif sampling_type == 'attentive':
                                        #pos_exp = tf.multiply(tf.exp(-logits), pos_mask)
                                        #pos_prob = tf.divide(pos_exp, tf.reduce_sum(pos_exp))
                                        neg_exp = tf.multiply(tf.exp(logits), neg_mask)
                                        neg_prob = tf.divide(neg_exp, tf.reduce_sum(neg_exp))

                                    #pos_sample_size = tf.cast(tf.ceil(tf.multiply(pos_sample_percent, pos_count)), tf.int32)
                                    #neg_sample_size = tf.cast(tf.ceil(tf.multiply(neg_sample_percent, neg_count)), tf.int32)
                                    #pos_indices = tf.py_func(np.random.choice, [input_shape, pos_sample_size, False,
                                    #                                            pos_prob], tf.int64)
                                    #pos_indices = tf.cast(pos_indices, tf.int32)
                                    # neg_indices = tf.py_func(np.random.choice, [input_shape, neg_sample_size, False,
                                    #                                             neg_prob], tf.int64)

                                    _, neg_indices = tf.nn.top_k(neg_prob, neg_sample_size, False)
                                    #indices = tf.concat([pos_indices, neg_indices], axis=0)
                                    #logits = tf.gather(logits, indices)
                                    #g1_matrix = tf.gather(g1_matrix, indices)
                                    #pos_mask = tf.gather(pos_mask, indices)
                                    neg_mask = tf.gather(neg_mask, neg_indices)
                                    neg_logits = tf.gather(logits, neg_indices)
                                    #pos_count = tf.reduce_sum(pos_mask)
                                    #neg_count = tf.reduce_sum(neg_count)

                                #pos_count_keep = tf.reduce_sum(pos_mask,axis=1, keep_dims=True)
                                    neg_exp = tf.exp(tf.multiply(neg_mask, neg_logits)) #[a]
                                    neg_exp = tf.multiply(neg_exp, neg_mask)
                                    neg_exp_sum = tf.reduce_sum(neg_exp) #[1]
                                    #avg_neg_exp_sum = tf.divide(neg_exp_sum, neg_count) #[q, 1]
                                    #less_than_box_sum = (float(max_answer_size) - tf.cast(self.answer_count, tf.float32)) * avg_neg_exp_sum #[q,1]
                                    #pos_effect_sum = tf.multiply(pos_count_keep-1, avg_neg_exp_sum) #[q,1]
                                    #neg_exp_sum = tf.add(neg_exp_sum, tf.add(less_than_box_sum, pos_effect_sum)) #[q, 1]
                                    pos_exp = tf.exp(tf.multiply(pos_mask, logits)) # [a]
                                    fi = tf.log(1 + tf.divide(neg_exp_sum, pos_exp)) #[a]
                                    fi = tf.multiply(fi, pos_mask) #[a]
                                    #fi = tf.reduce_sum(fi, axis=1) #[q]
                                    #fi = tf.divide(fi,pos_count) #[q]
                                    #self.loss = tf.reduce_mean(fi)

                                    #fi = tf.reduce_sum(fi) #[1]
                                    #self.loss = tf.divide(fi, pos_count_all) #[1]

                                    fi = tf.reduce_sum(fi) #[1]
                                    if pos_avg == True:
                                        fi = tf.divide(fi, pos_count) #[1]
                                        loss_list.append(fi)
                                    else:
                                        pos_list.append (pos_count)
                                        loss_list.append(fi)

                                else:
                                    neg_exp = tf.exp(tf.multiply(neg_mask, logits)) #[a]
                                    neg_exp = tf.multiply(neg_exp, neg_mask)
                                    neg_exp_sum = tf.reduce_sum(neg_exp) #[1]
                                    pos_exp = tf.exp(tf.multiply(pos_mask, logits)) # [a]
                                    fi = tf.log(1 + tf.divide(neg_exp_sum, pos_exp)) #[a]
                                    fi = tf.multiply(fi, pos_mask) #[a]
                                    #fi = tf.reduce_sum(fi, axis=1) #[q]
                                    #fi = tf.divide(fi,pos_count) #[q]
                                    #self.loss = tf.reduce_mean(fi)

                                    #fi = tf.reduce_sum(fi) #[1]
                                    #self.loss = tf.divide(fi, pos_count_all) #[1]

                                    fi = tf.reduce_sum(fi) #[1]
                                    if pos_avg == True:
                                        fi = tf.divide(fi, pos_count) #[1]
                                        loss_list.append(fi)
                                    else:
                                        pos_list.append (pos_count)
                                        loss_list.append(fi)
                            else:
                                hinget = tf.reshape(self.hinge_truth[i], [self.answer_count[i], self.answer_count[i]]) #[a, a]
                                loss_list.append(self.check_pairs(hinget, logits, top_treshold, pos_count))
                    else:
                        if with_tanh == True:
                            logits = tf.tanh(logits)
                        self.prob = tf.reshape(logits, [-1])
                        #self.loss = self.hinge_loss(g1_matrix, logits)
                        self.loss = self.hinge_loss(self.hinge_truth, logits)
                else:
                    logit_list = tf.unstack(logits,axis = 1 ,num=2)
                    score_list.append(logit_list[1])
                    prob_list.append(logit_list[1])
                    gold_matrix = self.truth[i]
                    g1_matrix = tf.ceil(gold_matrix - eps)
                    g1_matrix = tf.cast(g1_matrix + eps, tf.int32)
                    #self.prob = tf.nn.softmax(logits)
                    gold_matrix = tf.one_hot(g1_matrix, num_classes, dtype=tf.float32)            #         gold_matrix = tf.one_hot(self.truth, num_classes)
                    #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, gold_matrix))
                    loss_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix)))
        #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(self.truth, tf.int64), name='cross_entropy_per_example')
        #         self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
                    #correct = tf.nn.in_top_k(logits, self.truth, 1)
                    #self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                    #self.predictions = tf.arg_max(self.prob, 1)
                    #self.predictions = tf.argmax(self.prob, 1)

                tf.get_variable_scope().reuse_variables()

        #print (len (loss_list))
        self.loss = tf.stack (loss_list, 0)
        if pos_avg == True or pos_avg == False: #this change for #divcount
            self.loss = tf.reduce_mean(self.loss, 0)
        else:
            pos_cnt = tf.stack(pos_list, 0)
            pos_cnt = tf.reduce_sum(pos_cnt)
            self.loss = tf.reduce_sum(self.loss)
            self.loss = tf.divide(self.loss , pos_cnt)
        self.score = tf.concat(score_list, 0)
        self.prob = tf.concat(prob_list, 0)

        if optimize_type == 'adadelta':
            clipper = 50 
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 
        elif optimize_type == 'sgd':
            self.global_step = tf.Variable(0, name='global_step', trainable=False) # Create a variable to track the global step.
            min_lr = 0.000001
            self._lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
        elif optimize_type == 'ema':
            tvars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
            maintain_averages_op = ema.apply(tvars)
            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([train_op]):
                self.train_op = tf.group(maintain_averages_op)
        elif optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    def check_pairs (self, hinge_truth, logits, margine, pos_count):
        # only mask(pos, neg) instances are 1. the others: (pos, pos), (neg, neg), (neg, pos) are zero
        # g = x[0]
        mask = hinge_truth
        l = logits
        # g_p = tf.expand_dims(g, axis=0)
        # g_n = tf.expand_dims(g, axis=-1)
        # mask = tf.subtract(g_p, g_n)
        # mask = mask + 1
        # mask = mask // 2

        #compare each pair of neg and pos based on hinge loss
        l_p = tf.expand_dims(l, axis=0) #[1, a]
        l_n = tf.expand_dims(l, axis=-1) #[a, 1]
        sub1 = tf.subtract(l_p, l_n) - margine
        new_mask = tf.minimum(0.0, sub1) # hame +ha beshan 0
        new_mask = tf.maximum(-0.5, new_mask) # hame -ha bishtar az -0.5
        new_mask = -new_mask
        new_mask = tf.ceil(new_mask - eps)
        mask = tf.multiply(mask , new_mask)

        l_final = tf.exp(tf.subtract(l_n, l_p))
        l_final = tf.multiply(l_final, mask)
        l_final = tf.reduce_sum(l_final, axis=0)
        l_final = tf.reduce_sum(tf.log (1 + l_final))

        return tf.divide(l_final, pos_count)
            #return l_final

    def hinge_loss (self, hinge_truth, logits, soft_hinge=True):
        # hinge_loss:
        # loss = avg (max(0, 1 - (s+) + (s-))
        def singel_instance(x):
            # only mask(pos, neg) instances are 1. the others: (pos, pos), (neg, neg), (neg, pos) are zero
            # g = x[0]
            mask = x[0]
            l = x[1]
            # g_p = tf.expand_dims(g, axis=0)
            # g_n = tf.expand_dims(g, axis=-1)
            # mask = tf.subtract(g_p, g_n)
            # mask = mask + 1
            # mask = mask // 2

            #compare each pair of neg and pos based on hinge loss
            l_p = tf.expand_dims(l, axis=0)
            l_n = tf.expand_dims(l, axis=-1)
            if soft_hinge == False:
                l_final = tf.maximum(0.0, tf.subtract(l_n, l_p) + 1)
            else:
                l_final = tf.subtract(l_n, l_p)
            l_final = tf.multiply(l_final, mask)
            if soft_hinge == True:
                l_final = tf.log(1 + tf.exp(l_final))
                l_final = tf.multiply(l_final, mask)
            l_final = tf.reduce_sum(l_final)
            return tf.divide(l_final, mask) #[0]
            #return l_final

        elems = (hinge_truth, logits)
        return tf.reduce_mean(tf.map_fn(singel_instance, elems, dtype=tf.float32)) #[question_count] -> [0]

        #l_final = tf.map_fn(singel_instance, elems, dtype=tf.float32) #[q]
        #l_final = tf.reduce_sum(l_final)
        #mask = tf.reduce_sum(hinge_truth)
        #return tf.divide(l_final, mask)

    def get_hinge_truth(self):
        return self.__hinge_truth

    def set_hinge_truth(self, value):
        self.__hinge_truth = value

    def del_hinge_truth(self):
        del self.__hinge_truth


    def get_question_count(self):
        return self.__question_count

    def set_question_count(self, value):
        self.__question_count = value

    def del_question_count(self):
        del self.__question_count

    def get_answer_count(self):
        return self.__answer_count

    def set_answer_count(self, value):
        self.__answer_count = value

    def del_answer_count (self):
        del self.__answer_count

    def get_real_answer_count_mask(self):
        return self.__real_answer_count_mask
    def set_real_answer_count_mask(self, value):
        self.__real_answer_count_mask = value
    def del_real_answer_count_mask(self):
        del self.__real_answer_count_mask

    def get_overlap(self):
        return self.__overlap

    def set_overlap(self, value):
        self.__overlap = value

    def del_overlap(self):
        del self.__overlap

    def get_score(self):
        return self.__score

    def set_score(self, value):
        self.__score = value

    def del_score(self):
        del self.__score

    def get_predictions(self):
        return self.__predictions

    def set_predictions(self, value):
        self.__predictions = value


    def del_predictions(self):
        del self.__predictions



    def get_eval_correct(self):
        return self.__eval_correct


    def set_eval_correct(self, value):
        self.__eval_correct = value


    def del_eval_correct(self):
        del self.__eval_correct


    def get_question_lengths(self):
        return self.__question_lengths


    def get_passage_lengths(self):
        return self.__passage_lengths


    def get_truth(self):
        return self.__truth


    def get_in_question_words(self):
        return self.__in_question_words


    def get_in_passage_words(self):
        return self.__in_passage_words


    def get_word_embedding(self):
        return self.__word_embedding


    def get_in_question_poss(self):
        return self.__in_question_POSs


    def get_in_passage_poss(self):
        return self.__in_passage_POSs


    def get_pos_embedding(self):
        return self.__POS_embedding


    def get_in_question_ners(self):
        return self.__in_question_NERs


    def get_in_passage_ners(self):
        return self.__in_passage_NERs


    def get_ner_embedding(self):
        return self.__NER_embedding


    def get_question_char_lengths(self):
        return self.__question_char_lengths


    def get_passage_char_lengths(self):
        return self.__passage_char_lengths


    def get_in_question_chars(self):
        return self.__in_question_chars


    def get_in_passage_chars(self):
        return self.__in_passage_chars


    def get_char_embedding(self):
        return self.__char_embedding


    def get_prob(self):
        return self.__prob

    def get_attention_weights (self):
        return self.__attention_weights


    def get_prediction(self):
        return self.__prediction


    def get_loss(self):
        return self.__loss


    def get_train_op(self):
        return self.__train_op


    def get_global_step(self):
        return self.__global_step


    def get_lr_rate(self):
        return self.__lr_rate


    def set_question_lengths(self, value):
        self.__question_lengths = value


    def set_passage_lengths(self, value):
        self.__passage_lengths = value


    def set_truth(self, value):
        self.__truth = value


    def set_in_question_words(self, value):
        self.__in_question_words = value


    def set_in_passage_words(self, value):
        self.__in_passage_words = value


    def set_word_embedding(self, value):
        self.__word_embedding = value


    def set_in_question_poss(self, value):
        self.__in_question_POSs = value


    def set_in_passage_poss(self, value):
        self.__in_passage_POSs = value


    def set_pos_embedding(self, value):
        self.__POS_embedding = value


    def set_in_question_ners(self, value):
        self.__in_question_NERs = value


    def set_in_passage_ners(self, value):
        self.__in_passage_NERs = value


    def set_ner_embedding(self, value):
        self.__NER_embedding = value


    def set_question_char_lengths(self, value):
        self.__question_char_lengths = value


    def set_passage_char_lengths(self, value):
        self.__passage_char_lengths = value


    def set_in_question_chars(self, value):
        self.__in_question_chars = value


    def set_in_passage_chars(self, value):
        self.__in_passage_chars = value


    def set_char_embedding(self, value):
        self.__char_embedding = value


    def set_prob(self, value):
        self.__prob = value

    def set_attention_weights (self, value):
        self.__attention_weights = value


    def set_prediction(self, value):
        self.__prediction = value


    def set_loss(self, value):
        self.__loss = value


    def set_train_op(self, value):
        self.__train_op = value


    def set_global_step(self, value):
        self.__global_step = value


    def set_lr_rate(self, value):
        self.__lr_rate = value


    def del_question_lengths(self):
        del self.__question_lengths


    def del_passage_lengths(self):
        del self.__passage_lengths


    def del_truth(self):
        del self.__truth


    def del_in_question_words(self):
        del self.__in_question_words


    def del_in_passage_words(self):
        del self.__in_passage_words


    def del_word_embedding(self):
        del self.__word_embedding


    def del_in_question_poss(self):
        del self.__in_question_POSs


    def del_in_passage_poss(self):
        del self.__in_passage_POSs


    def del_pos_embedding(self):
        del self.__POS_embedding


    def del_in_question_ners(self):
        del self.__in_question_NERs


    def del_in_passage_ners(self):
        del self.__in_passage_NERs


    def del_ner_embedding(self):
        del self.__NER_embedding


    def del_question_char_lengths(self):
        del self.__question_char_lengths


    def del_passage_char_lengths(self):
        del self.__passage_char_lengths


    def del_in_question_chars(self):
        del self.__in_question_chars


    def del_in_passage_chars(self):
        del self.__in_passage_chars


    def del_char_embedding(self):
        del self.__char_embedding


    def del_prob(self):
        del self.__prob

    def del_attention_weights(self):
        del self.__attention_weights

    def del_prediction(self):
        del self.__prediction


    def del_loss(self):
        del self.__loss


    def del_train_op(self):
        del self.__train_op


    def del_global_step(self):
        del self.__global_step


    def del_lr_rate(self):
        del self.__lr_rate

    question_lengths = property(get_question_lengths, set_question_lengths, del_question_lengths, "question_lengths's docstring")
    passage_lengths = property(get_passage_lengths, set_passage_lengths, del_passage_lengths, "passage_lengths's docstring")
    truth = property(get_truth, set_truth, del_truth, "truth's docstring")
    in_question_words = property(get_in_question_words, set_in_question_words, del_in_question_words, "in_question_words's docstring")
    in_passage_words = property(get_in_passage_words, set_in_passage_words, del_in_passage_words, "in_passage_words's docstring")
    word_embedding = property(get_word_embedding, set_word_embedding, del_word_embedding, "word_embedding's docstring")
    in_question_POSs = property(get_in_question_poss, set_in_question_poss, del_in_question_poss, "in_question_POSs's docstring")
    in_passage_POSs = property(get_in_passage_poss, set_in_passage_poss, del_in_passage_poss, "in_passage_POSs's docstring")
    POS_embedding = property(get_pos_embedding, set_pos_embedding, del_pos_embedding, "POS_embedding's docstring")
    in_question_NERs = property(get_in_question_ners, set_in_question_ners, del_in_question_ners, "in_question_NERs's docstring")
    in_passage_NERs = property(get_in_passage_ners, set_in_passage_ners, del_in_passage_ners, "in_passage_NERs's docstring")
    NER_embedding = property(get_ner_embedding, set_ner_embedding, del_ner_embedding, "NER_embedding's docstring")
    question_char_lengths = property(get_question_char_lengths, set_question_char_lengths, del_question_char_lengths, "question_char_lengths's docstring")
    passage_char_lengths = property(get_passage_char_lengths, set_passage_char_lengths, del_passage_char_lengths, "passage_char_lengths's docstring")
    in_question_chars = property(get_in_question_chars, set_in_question_chars, del_in_question_chars, "in_question_chars's docstring")
    in_passage_chars = property(get_in_passage_chars, set_in_passage_chars, del_in_passage_chars, "in_passage_chars's docstring")
    char_embedding = property(get_char_embedding, set_char_embedding, del_char_embedding, "char_embedding's docstring")
    prob = property(get_prob, set_prob, del_prob, "prob's docstring")
    attention_weights = property(get_attention_weights, set_attention_weights, del_attention_weights, "prob's docstring")
    prediction = property(get_prediction, set_prediction, del_prediction, "prediction's docstring")
    loss = property(get_loss, set_loss, del_loss, "loss's docstring")
    train_op = property(get_train_op, set_train_op, del_train_op, "train_op's docstring")
    global_step = property(get_global_step, set_global_step, del_global_step, "global_step's docstring")
    lr_rate = property(get_lr_rate, set_lr_rate, del_lr_rate, "lr_rate's docstring")
    eval_correct = property(get_eval_correct, set_eval_correct, del_eval_correct, "eval_correct's docstring")
    predictions = property(get_predictions, set_predictions, del_predictions, "predictions's docstring")
    question_count = property(get_question_count, set_question_count, del_question_count, "predictions's docstring")
    answer_count = property(get_answer_count, set_answer_count, del_answer_count, "predictions's docstring")
    overlap = property(get_overlap, set_overlap, del_overlap, "asdfas")
    score = property(get_score, set_score, del_score, "asdfasdfa")
    hinge_truth = property(get_hinge_truth, set_hinge_truth, del_hinge_truth, "asdfasdfa")
    real_answer_count_mask = property(get_real_answer_count_mask, set_real_answer_count_mask,
                                      del_real_answer_count_mask, "asdfasdfa")

    
