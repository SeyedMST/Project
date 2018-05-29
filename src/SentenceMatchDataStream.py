import numpy as np
import re
import random

import sys
eps = 1e-8

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def make_batches_as (instances, batch_size, min_random_param, is_training):
    ans = []
    ans_len = []
    question_count = []
    last_size = 0
    ss = 0
    ss_tot = 0
    start = 0
    count = 0
    for x in instances:
        if ss_tot == 0:
            last_size = len(x[1])
        if (len(x[1]) == last_size and ss + len(x[1]) <= batch_size):
            ss += len(x[1])
        else:
            ans.append((start, ss_tot))
            ans_len.append(last_size)
            question_count.append(count)
            count = 0
            last_size = len(x[1])
            start = ss_tot
            ss = len (x[1])
        ss_tot += len(x[1])
        count += 1
    ans.append((start, ss_tot))
    ans_len.append(last_size)
    question_count.append(count)
    return (ans, question_count, ans_len)


def pad_2d_matrix(in_val, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in range(batch_size):
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val

def pad_3d_tensor(in_val, max_length1=None, max_length2=None, dtype=np.int32):
    #print('x')
    if max_length1 is None: max_length1 = np.max([len(cur_in_val) for cur_in_val in in_val])
    if max_length2 is None: max_length2 = np.max([np.max([len(val) for val in cur_in_val]) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length1, max_length2), dtype=dtype)
    for i in range(batch_size):
        cur_length1 = max_length1
        if len(in_val[i])<max_length1: cur_length1 = len(in_val[i])
        for j in range(cur_length1):
            cur_in_val = in_val[i][j]
            kept_length = len(cur_in_val)
            if kept_length>max_length2: kept_length = max_length2
            out_val[i, j, :kept_length] = cur_in_val[:kept_length]
    #print('y')
    return out_val


def wikiQaGenerate(filename, label_vocab, word_vocab, char_vocab, max_sent_length, batch_size, is_training, is_list_wise):

    is_trec = False
    if 'trec' in filename:
        is_trec = True
    if is_training == True and is_list_wise == True:
        if is_trec == True:
            min_answer_size = 15
            max_answer_size = 60
        else:
            min_answer_size = 15
            max_answer_size = 15
    elif is_training == True:
        max_answer_size = 79
        min_answer_size = 0
    else:
        max_answer_size = 20000
        min_answer_size = 0
        #print ("hahaha")
    data = open(filename, 'rt')
    question_dic = {}
    negative_answers = []
    question_count = 0 #wiki 2,118
    all_count = 0 #wiki 20,360
    del_question_count = 0 #wiki 1,245 (59% of questions deleted, 873 question remaine)
    del_all_count = 0 #wiki 11,688 (57% of pairs deleted, 8,672 remaine(9.9 answer per question))

    for line in data:
        if sys.version_info[0] < 3:
            line = line.decode('utf-8').strip()
        else:
            line = line.strip()
        #if line.startswith('-'): continue
        item = re.split("\t", line)
        question_dic.setdefault(str(item[0]),{})
        question_dic[str(item[0])].setdefault("question",[])
        question_dic[str(item[0])].setdefault("answer",[])
        question_dic[str(item[0])].setdefault("label",[])
        question_dic[str(item[0])]["question"].append(item[0].lower())
        question_dic[str(item[0])]["answer"].append(item[1].lower())
        question_dic[str(item[0])]["label"].append(int(item[2]))
        if int(item[2]) == 0:
            negative_answers.append(item[1].lower())
        all_count += 1
    data.close()
    #for key in question_dic.keys():
    for key in list(question_dic):
        question_count += 1
        question_dic[key]["question"] = question_dic[key]["question"]
        question_dic[key]["answer"] = question_dic[key]["answer"]
        if sum(question_dic[key]["label"]) <eps or (is_training == True and len(question_dic[key]["question"]) == sum(question_dic[key]["label"])):
            del_question_count += 1
            del_all_count += len(question_dic[key]["question"])
            del(question_dic[key])
        elif is_trec == True and len(question_dic[key]["question"]) == sum(question_dic[key]["label"]):
            del_question_count += 1
            del_all_count += len(question_dic[key]["question"])
            del(question_dic[key])

    question = list()
    answer = list()
    label = list()
    pairs_count = 0
    for item in question_dic.values():
        good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1]
        good_length = len(good_answer)
        bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0]
        if len(item["answer"]) > max_answer_size:
            good_answer.extend(random.sample(bad_answer,max_answer_size - good_length))
            temp_answer = good_answer
            temp_label = [1 / float(sum(item["label"])) for i in range(good_length)]
            temp_label.extend([0.0 for i in range(max_answer_size-good_length)])
        else:
            temp_answer = item["answer"]
            temp_label = [x / float(sum(item["label"])) for x in item["label"]]
            if min_answer_size-len(item["question"]) >= 1:
                temp_answer.extend(random.sample(negative_answers, min_answer_size-len(item["question"])))
                temp_label.extend([0.0 for i in range(min_answer_size-len(item["question"]))])
        label.append(temp_label) # label[i] = list of labels of question i
        answer.append(temp_answer) # answer[i] = list of answers of question i
        question += [([item["question"][0]])[0]] # question[i] = question i
        pairs_count += len(temp_answer)
    question = np.array(question) # list of questions
    answer = np.array(answer) # list of list of answers
    label = np.array(label) #list of list of labels

    instances = []
    for i in range(len(question)):
        instances.append((question[i], answer[i], label[i]))
    instances = sorted(instances, key=lambda instance: (len(instance[1]))) #sort based on len (answer[i])
    if is_training == True:
        batches = make_batches_as(instances, batch_size, min_answer_size, is_training)
    else:
        batches = make_batches(pairs_count,batch_size)
    ans = []
    candidate_answer_length = []
    for x in (instances):
        candidate_answer_length.append(len(x[1]))
        for j in range (len(x[1])):
            label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2 = \
                make_idx(label_vocab, x[2][j], word_vocab, x[0], x[1][j], char_vocab, max_sent_length, True)
            my_label = '0'
            if x[2][j] > 0.0001:
                my_label = '1'
            ans.append(
                (my_label, x[0], x[1][j], label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                 None, None, None, None))
    print ("Questions: ",len(instances), " pairs: ", len(ans))
    return (ans, batches, candidate_answer_length)


def make_idx (label_vocab, label, word_vocab, sentence1, sentence2, char_vocab, max_sent_length, is_as):
    if is_as == False:
        if label_vocab is not None:
            label_id = label_vocab.getIndex(label)
            if label_id >= label_vocab.vocab_size: label_id = 0
        else:
            label_id = int(label)
    else:
        label_id = label # 1/len(goood_answers)
    word_idx_1 = word_vocab.to_index_sequence(sentence1)
    word_idx_2 = word_vocab.to_index_sequence(sentence2)
    char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
    char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)
    if len(word_idx_1) > max_sent_length:
        word_idx_1 = word_idx_1[:max_sent_length]
        char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
    if len(word_idx_2) > max_sent_length:
        word_idx_2 = word_idx_2[:max_sent_length]
        char_matrix_idx_2 = char_matrix_idx_2[:max_sent_length]
    return (label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2)


import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
from nltk.corpus import wordnet
import string

# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

def get_wordnet_pos(pos_tag):
    if pos_tag[1].startswith('J'):
        return (pos_tag[0], wordnet.ADJ)
    elif pos_tag[1].startswith('V'):
        return (pos_tag[0], wordnet.VERB)
    elif pos_tag[1].startswith('N'):
        return (pos_tag[0], wordnet.NOUN)
    elif pos_tag[1].startswith('R'):
        return (pos_tag[0], wordnet.ADV)
    else:
        return (pos_tag[0], wordnet.NOUN)

# Create tokenizer and stemmer
tokenizer = nltk.tokenize.SpaceTokenizer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def is_ci_lemma_stopword_set_match(sentence1, sentence2):
    """Check if a and b are matches."""
    s1 = []
    s2 = []
    for word in re.split('\\s+', sentence1):
        s1.append(word)
    for word in re.split('\\s+', sentence2):
        s2.append(word)
    pos_a = map(get_wordnet_pos, nltk.pos_tag(s1))#(tokenizer.tokenize(sentence1, ' ')))
    pos_b = map(get_wordnet_pos, nltk.pos_tag(s2))#(tokenizer.tokenize(sentence2, ' ')))
    lemmae_a = [lemmatizer.lemmatize(token.strip(), pos) for token, pos in pos_a]
    lemmae_b = [lemmatizer.lemmatize(token.strip(), pos) for token, pos in pos_b]
    return lemmae_a, lemmae_b

def add_overlap (sentence1_list, sentence2_list, sentence1, sentence2, word_vocab, is_word_overlap, is_lemma_overlap):
    n = len(sentence1_list)
    m = len(sentence2_list)
    if is_word_overlap == True and is_lemma_overlap == True:
        lemma1, lemma2 = is_ci_lemma_stopword_set_match(sentence1, sentence2)
    # if (len(lemma1) != n or len (lemma2) != m):
    #     print ('fuck')
    ans = []
    for i in range(n):
        l = []
        for j in range(m):
            if is_word_overlap == True or is_lemma_overlap == True:
                print ('fuck')
                if is_lemma_overlap == False and sentence1_list[i] == sentence2_list[j]:
                    l.append(1)
                elif is_lemma_overlap == True:
                    if sentence1_list[i] == sentence2_list[j] or lemma1[i] == lemma2[j]:
                        l.append(1)
                    else:
                        l.append(0)
                #     s1 = word_vocab.getWord (sentence1[i])
                #     s2 = word_vocab.getWord (sentence2[j])
                #     #s1 = wordnet.lemma(s1)
                #     #s2 = wordnet.lemma(s2)
                #     if (s1 == s2):
                #         l.append(1)
                #     elif s1 != None and s2 != None:
                #         ss = wordnet.synsets ('ChanGe')
                #         print (ss)
                #         syn1 = wordnet.synsets(s1)
                #         syn2 = wordnet.synsets(s2)
                #         for x in syn1:
                #             for y in syn2:
                #                 if x == y:
                #                     l.append(1)
                #                     break
                #         l.append(0)
                else:
                    l.append(0)
            else:
                l.append(0)
        ans.append(l)
    return ans


class SentenceMatchDataStream(object):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, label_vocab=None, batch_size=60, 
                 isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, max_sent_length=200, is_as = True,
                 is_word_overlap = True, is_lemma_overlap = True, is_list_wise = False):
        instances = []
        batch_spans = []
        self.batch_as_len = []
        self.batch_question_count = []
        self.candidate_answer_length = []
        if (is_as == True):
            instances, r, self.candidate_answer_length = wikiQaGenerate(inpath,label_vocab, word_vocab, char_vocab, max_sent_length, batch_size,
                                          is_training=isShuffle, is_list_wise=is_list_wise)
            if isShuffle == True:
                batch_spans = r[0]
                self.batch_question_count = r[1]
                self.batch_as_len = r[2]
            else:
                batch_spans = r
                self.batch_question_count = 0
                self.batch_as_len = 0
        else:
            infile = open(inpath, 'rt')
            for line in infile:
                if sys.version_info[0] < 3:
                    line = line.decode('utf-8').strip()
                else:
                    line = line.strip()
                #if line.startswith('-'): continue
                items = re.split("\t", line)
                label = items[2]
                sentence1 = items[0].lower()
                sentence2 = items[1].lower()
                label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2 = \
                    make_idx(label_vocab, label, word_vocab, sentence1, sentence2, char_vocab, max_sent_length, is_as=False)
                POS_idx_1 = None
                POS_idx_2 = None
                if POS_vocab is not None:
                    POS_idx_1 = POS_vocab.to_index_sequence(items[3])
                    if len(POS_idx_1) > max_sent_length: POS_idx_1 = POS_idx_1[:max_sent_length]
                    POS_idx_2 = POS_vocab.to_index_sequence(items[4])
                    if len(POS_idx_2) > max_sent_length: POS_idx_2 = POS_idx_2[:max_sent_length]

                NER_idx_1 = None
                NER_idx_2 = None
                if NER_vocab is not None:
                    NER_idx_1 = NER_vocab.to_index_sequence(items[5])
                    if len(NER_idx_1) > max_sent_length: NER_idx_1 = NER_idx_1[:max_sent_length]
                    NER_idx_2 = NER_vocab.to_index_sequence(items[6])
                    if len(NER_idx_2) > max_sent_length: NER_idx_2 = NER_idx_2[:max_sent_length]


                instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                                  POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2))
            infile.close()
            # sort instances based on sentence length
            if isSort: instances = sorted(instances, key=lambda instance: (-len(instance[4])))#, len(instance[5]))) # sort instances based on length
            self.num_instances = len(instances)
            # distribute into different buckets
            batch_spans = make_batches(self.num_instances, batch_size)

        self.num_instances = len(instances)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            char_matrix_idx_1_batch = []
            char_matrix_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []
            sent1_char_length_batch = []
            sent2_char_length_batch = []

            overlap_batch = []

            POS_idx_1_batch = None
            if POS_vocab is not None: POS_idx_1_batch = []
            POS_idx_2_batch = None
            if POS_vocab is not None: POS_idx_2_batch = []

            NER_idx_1_batch = None
            if NER_vocab is not None: NER_idx_1_batch = []
            NER_idx_2_batch = None
            if NER_vocab is not None: NER_idx_2_batch = []

            for i in range(batch_start, batch_end):
                (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                 POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2) = instances[i]
                label_batch.append(label)
                sent1_batch.append(sentence1)
                sent2_batch.append(sentence2)
                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                char_matrix_idx_1_batch.append(char_matrix_idx_1)
                char_matrix_idx_2_batch.append(char_matrix_idx_2)
                sent1_length_batch.append(len(word_idx_1))
                sent2_length_batch.append(len(word_idx_2))
                sent1_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                sent2_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])

                overlap_batch.append(add_overlap(word_idx_1,word_idx_2,sentence1, sentence2, word_vocab, is_word_overlap,
                                                 is_lemma_overlap))
                if POS_vocab is not None:
                    POS_idx_1_batch.append(POS_idx_1)
                    POS_idx_2_batch.append(POS_idx_2)

                if NER_vocab is not None: 
                    NER_idx_1_batch.append(NER_idx_1)
                    NER_idx_2_batch.append(NER_idx_2)
                
            cur_batch_size = len(label_batch)
            if cur_batch_size ==0: continue

            # padding
            max_sent1_length = np.max(sent1_length_batch)
            max_sent2_length = np.max(sent2_length_batch)
            max_char_length1 = np.max([np.max(aa) for aa in sent1_char_length_batch])
            if max_char_length1>max_char_per_word: max_char_length1=max_char_per_word

            max_char_length2 = np.max([np.max(aa) for aa in sent2_char_length_batch])
            if max_char_length2>max_char_per_word: max_char_length2=max_char_per_word
            
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)

            char_matrix_idx_1_batch = pad_3d_tensor(char_matrix_idx_1_batch, max_length1=max_sent1_length, max_length2=max_char_length1)
            char_matrix_idx_2_batch = pad_3d_tensor(char_matrix_idx_2_batch, max_length1=max_sent2_length, max_length2=max_char_length2)

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            sent1_char_length_batch = pad_2d_matrix(sent1_char_length_batch, max_length=max_sent1_length)
            sent2_char_length_batch = pad_2d_matrix(sent2_char_length_batch, max_length=max_sent2_length)

            overlap_batch = pad_3d_tensor(overlap_batch, max_length1=max_sent1_length, max_length2=max_sent2_length)


            if POS_vocab is not None:
                POS_idx_1_batch = pad_2d_matrix(POS_idx_1_batch, max_length=max_sent1_length)
                POS_idx_2_batch = pad_2d_matrix(POS_idx_2_batch, max_length=max_sent2_length)
            if NER_vocab is not None:
                NER_idx_1_batch = pad_2d_matrix(NER_idx_1_batch, max_length=max_sent1_length)
                NER_idx_2_batch = pad_2d_matrix(NER_idx_2_batch, max_length=max_sent2_length)
                

            self.batches.append((label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch,
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, overlap_batch))
        
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
        print ('finish')


    def answer_count (self, i):
        return self.batch_as_len[i]

    def question_count (self, i):
        return self.batch_question_count[i]

    def get_candidate_answer_length (self):
        return self.candidate_answer_length

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch, self.index_array[self.cur_pointer-1]

    def reset(self):
        #if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]
        
