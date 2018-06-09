import tensorflow as tf
from tensorflow.python.ops import rnn
#import my_rnn

eps = 1e-8
# def cosine_distance(y1,y2):
#     # y1 [....,a, 1, d]
#     # y2 [....,1, b, d]
# #     cosine_numerator = T.sum(y1*y2, axis=-1)
#     cosine_numerator = tf.reduce_sum(tf.mul(y1, y2), axis=-1)
# #     y1_norm = T.sqrt(T.maximum(T.sum(T.sqr(y1), axis=-1), eps)) #be careful while using T.sqrt(), like in the cases of Euclidean distance, cosine similarity, for the gradient of T.sqrt() at 0 is undefined, we should add an Eps or use T.maximum(original, eps) in the sqrt.
#     y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
#     y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
#     return cosine_numerator / y1_norm / y2_norm
#
# def cal_relevancy_matrix(in_question_repres, in_passage_repres):
#     in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
#     in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
#     relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
#     return relevancy_matrix
#
# def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
#     # relevancy_matrix: [batch_size, passage_len, question_len]
#     # question_mask: [batch_size, question_len]
#     # passage_mask: [batch_size, passsage_len]
#     relevancy_matrix = tf.mul(relevancy_matrix, tf.expand_dims(question_mask, 1))
#     relevancy_matrix = tf.mul(relevancy_matrix, tf.expand_dims(passage_mask, 2))
#     return relevancy_matrix
#
# def cal_cosine_weighted_question_representation(question_representation, cosine_matrix, normalize=False):
#     # question_representation: [batch_size, question_len, dim]
#     # cosine_matrix: [batch_size, passage_len, question_len]
#     if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
#     expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, passage_len, question_len, 'x']
#     weighted_question_words = tf.expand_dims(question_representation, axis=1) # [batch_size, 'x', question_len, dim]
#     weighted_question_words = tf.reduce_sum(tf.mul(weighted_question_words, expanded_cosine_matrix), axis=2)# [batch_size, passage_len, dim]
#     if not normalize:
#         weighted_question_words = tf.div(weighted_question_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
#     return weighted_question_words # [batch_size, passage_len, dim]
#
# def multi_perspective_expand_for_3D(in_tensor, decompose_params):
#     in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, passage_len, 'x', dim]
#     decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
#     return tf.mul(in_tensor, decompose_params)#[batch_size, passage_len, decompse_dim, dim]
#
# def multi_perspective_expand_for_2D(in_tensor, decompose_params):
#     in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
#     decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
#     return tf.mul(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]
#
# def multi_perspective_expand_for_1D(in_tensor, decompose_params):
#     in_tensor = tf.expand_dims(in_tensor, axis=0) #['x', dim]
#     return tf.mul(in_tensor, decompose_params) # [decompse_dim, dim]
#
#
# def cal_full_matching_bak(passage_representation, full_question_representation, decompose_params):
#     # passage_representation: [batch_size, passage_len, dim]
#     # full_question_representation: [batch_size, dim]
#     # decompose_params: [decompose_dim, dim]
#     mp_passage_rep = multi_perspective_expand_for_3D(passage_representation, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
#     mp_full_question_rep = multi_perspective_expand_for_2D(full_question_representation, decompose_params) # [batch_size, decompse_dim, dim]
#     return cosine_distance(mp_passage_rep, tf.expand_dims(mp_full_question_rep, axis=1)) #[batch_size, passage_len, decompse_dim]
#
# def cal_full_matching(passage_representation, full_question_representation, decompose_params):
#     # passage_representation: [batch_size, passage_len, dim]
#     # full_question_representation: [batch_size, dim]
#     # decompose_params: [decompose_dim, dim]
#     def singel_instance(x):
#         p = x[0]
#         q = x[1]
#         # p: [pasasge_len, dim], q: [dim]
#         p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
#         q = multi_perspective_expand_for_1D(q, decompose_params) # [decompose_dim, dim]
#         q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
#         return cosine_distance(p, q) # [passage_len, decompose]
#     elems = (passage_representation, full_question_representation)
#     return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]
#
# def cal_maxpooling_matching_bak(passage_rep, question_rep, decompose_params):
#     # passage_representation: [batch_size, passage_len, dim]
#     # qusetion_representation: [batch_size, question_len, dim]
#     # decompose_params: [decompose_dim, dim]
#     passage_rep = multi_perspective_expand_for_3D(passage_rep, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
#     question_rep = multi_perspective_expand_for_3D(question_rep, decompose_params) # [batch_size, question_len, decompse_dim, dim]
#
#     passage_rep = tf.expand_dims(passage_rep, 2) # [batch_size, passage_len, 1, decompse_dim, dim]
#     question_rep = tf.expand_dims(question_rep, 1) # [batch_size, 1, question_len, decompse_dim, dim]
#     matching_matrix = cosine_distance(passage_rep,question_rep) # [batch_size, passage_len, question_len, decompse_dim]
#     return tf.concat(2, [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]
#
# def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
#     # passage_representation: [batch_size, passage_len, dim]
#     # qusetion_representation: [batch_size, question_len, dim]
#     # decompose_params: [decompose_dim, dim]
#
#     def singel_instance(x):
#         p = x[0]
#         q = x[1]
#         # p: [pasasge_len, dim], q: [question_len, dim]
#         p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
#         q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
#         p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
#         q = tf.expand_dims(q, 0) # [1, question_len, decompose_dim, dim]
#         return cosine_distance(p, q) # [passage_len, question_len, decompose]
#     elems = (passage_rep, question_rep)
#     matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, decompse_dim]
#     return tf.concat(2, [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]
#
# def cal_maxpooling_matching_for_word(passage_rep, question_rep, decompose_params):
#     # passage_representation: [batch_size, passage_len, dim]
#     # qusetion_representation: [batch_size, question_len, dim]
#     # decompose_params: [decompose_dim, dim]
#
#     def singel_instance(x):
#         p = x[0]
#         q = x[1]
#         q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
#         # p: [pasasge_len, dim], q: [question_len, dim]
#         def single_instance_2(y):
#             # y: [dim]
#             y = multi_perspective_expand_for_1D(y, decompose_params) #[decompose_dim, dim]
#             y = tf.expand_dims(y, 0) # [1, decompose_dim, dim]
#             matching_matrix = cosine_distance(y, q)#[question_len, decompose_dim]
#             return tf.concat(0, [tf.reduce_max(matching_matrix, axis=0), tf.reduce_mean(matching_matrix, axis=0)]) #[2*decompose_dim]
#         return tf.map_fn(single_instance_2, p, dtype=tf.float32) # [passage_len, 2*decompse_dim]
#     elems = (passage_rep, question_rep)
#     return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, 2*decompse_dim]
#
#
# def cal_attentive_matching(passage_rep, att_question_rep, decompose_params):
#     # passage_rep: [batch_size, passage_len, dim]
#     # att_question_rep: [batch_size, passage_len, dim]
#     def singel_instance(x):
#         p = x[0]
#         q = x[1]
#         # p: [pasasge_len, dim], q: [pasasge_len, dim]
#         p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
#         q = multi_perspective_expand_for_2D(q, decompose_params) # [pasasge_len, decompose_dim, dim]
#         return cosine_distance(p, q) # [pasasge_len, decompose_dim]
#
#     elems = (passage_rep, att_question_rep)
#     return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]

def bilinear_att (passage_rep, question_rep, w, b):
    # w [d,d]; b[d]
    # r = q*w + b
    # gt = row_softmax(p*rT)
    # h = gt*q
    def single_instance (x):
        q = x[0] #[N,d]
        p = x[1] #[M,d]
        r = tf.nn.xw_plus_b(q, w, b) #[N, d]
        r = tf.transpose(r) #[d, N]
        gt = tf.nn.softmax(tf.matmul(p,r)) # [M, N]
        return tf.matmul(gt, q) #[M,d]
    elems = (question_rep, passage_rep)
    return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]

# def my_att_base (p, q, w):
#     ww = tf.expand_dims(w, 0)  # [1, d]
#     qq = tf.multiply(q, ww)  # [N,d]
#     qq = tf.expand_dims(qq, 0)  # [1, N, d]
#     pp = tf.expand_dims(p, 1)  # [M, 1, d]
#     z = tf.multiply(qq, pp)  # [M, N, d]
#     z = tf.reduce_sum(z, -1)  # [M, N]
#     return z

def dot_att_weight(passage_rep, question_rep,input_dim , overlap, question_mask, clip_att = True):
    #q_mask = tf.multiply((question_mask -1), 1000) #[bs, N]
    #q_mask = tf.expand_dims(q_mask, -1) #[bs, N, 1]
    q = question_rep #+ q_mask #[bs, N, d]
    p = tf.expand_dims(passage_rep, 2) #[bs, M, 1, d]
    q = tf.expand_dims(q, 1) #[bs, 1, N, d]
    z = tf.multiply(p, q) #[bs, M, N, d]
    #question_len = tf.reduce_sum(question_mask, -1) #[bs]
    #z2 = tf.reduce_sum(z, -1)#[bs, M, N]
    #z2 = tf.nn.softmax(z2) #[bs, M, N]

    def single_instance (x):
        z1 = x[0] #[M,N,d]
        q_mask1 = x[1] #[N]
        o1 = x[2] #[M,N]
        o1 = tf.expand_dims(o1, -1) #[M,N,1]
        o1 = tf.multiply(o1, z1) #[M,N,d]
        o1 = cal_wxb(o1,scope='overlap',output_dim=1,input_dim=input_dim,activation='relu') #[M,N,1]
        #o1 = tf.reduce_sum(o1, -1) #[M,N]

        # if clip_att == True:
        #     #z_c = cal_wxb(z1, scope='clip', output_dim=2, input_dim=input_dim) #[M,N,2]
        #     #z_c = tf.arg_max(input=z_c, dimension=2) #[M,N]
        #     z1 = tf.reduce_sum(z1, -1, keep_dims=True) #[M,N,1]
        #     z1 = z1+o1 #[M,N,1]
        #     z_c = cal_wxb(z1, scope='clip', output_dim=1, input_dim=1, activation = 'tanh') #[M,N,1]
        #     z_c = tf.reduce_sum(z_c, 2) #[M, N] Just for removing the last dimension
        #     z_c = tf.ceil(z_c) #[M,N]
        #     #z_c = tf.arg_max(input=z_c, dimension=2) #[M,N]
        #     #z_c = tf.to_float(z_c)
        #     z1 = tf.reduce_sum(z1, -1) #[M,N]
        #     z1 = tf.multiply(z1, z_c) #[M,N]
        # else:
        z1 = tf.reduce_sum(z1, -1, keep_dims=True) # [M,N,1]
        #z1 = z1 + o1
        z1 = tf.reduce_sum(z1, -1) #[M,N]

        #z1 = tf.nn.softmax(z1) #[M, N]
        z1_exp = tf.exp(z1) #[M,N]
        z1_exp = tf.multiply(z1_exp, tf.expand_dims(q_mask1, 0))  # [M,N]
        z1 = tf.divide(z1_exp,tf.reduce_sum(z1_exp, -1, keep_dims=True))
        #z1 = tf.multiply(z1, tf.expand_dims(q_mask1,0)) #[M,N]

        #z1_s = tf.reduce_sum(z1, -1, keep_dims=True) #[M, 1]
        #z1 = tf.divide(z1, z1_s) #[M,N]
        #print ('fuck you')
        if clip_att == True:
        #    print ('fuck me')
            # z1 = tf.expand_dims(z1, -1) #[M,N,1]
            # z_c = cal_wxb(z1, scope='clip', output_dim=1, input_dim=1, activation = 'tanh') #[M,N,1]
            # z_c = tf.reduce_sum(z_c, 2)  # [M, N] Just for removing the last dimension
            # z_c = tf.ceil(z_c)  # [M,N]
            # z1 = tf.reduce_sum(z1, -1)  # [M,N]
            # z1 = tf.multiply(z1, z_c)  #
            #z1_s = tf.reduce_sum(z1, -1, keep_dims=True) + eps #[M,1]
            #z1 = tf.divide(z1, z1_s) #[M, N]

            zr = z1 - 0.08
            zr = tf.ceil(zr)
            z1_exp = tf.multiply(z1_exp, zr)
            z1 = tf.divide(z1_exp,tf.reduce_sum(z1_exp, -1, keep_dims=True) + eps)


        return z1 #[M,N]
        #return tf.matmul(z1, q1) #[M,d]
    elems = (z, question_mask, overlap)
    return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]


def dot_product(passage_rep, question_rep,input_dim , overlap, question_mask,clip_attention):
    zz = dot_att_weight(passage_rep, question_rep,input_dim, overlap, question_mask, clip_attention) #[bs, M, N]
    def single_instance (x):
        z1 = x[0] #[M,N]
        q1 = x[1] #[N,d]
        return tf.matmul(z1, q1) #[M,d]
    elems = (zz, question_rep)
    return zz, tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]

def linear (passage_rep, question_rep, w, b):
    # def single_instance (x):
    #     q = x[0] #[N,d]
    #     p = x[1] #[M,d]
    #     z = my_att_base(p, q, w)
    #     z = tf.nn.softmax(z)  # [M,N]
    #     return tf.matmul(z, q) #[M,d]
    # elems = (question_rep, passage_rep)
    # return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]
    p_shape = tf.shape(passage_rep)
    batch_size = p_shape[0]
    passage_len = p_shape[1]
    dim = p_shape[2]
    p_shape = tf.shape(question_rep)
    quetion_len = p_shape[1]

    p = tf.expand_dims(passage_rep, 2) #[bs, M, 1, d]
    q = tf.expand_dims(question_rep, 1) #[bs, 1, N, d]
    z = tf.multiply(p, q) #[bs, M, N, d]
    z = tf.reshape(z, [-1, dim])
    z = tf.nn.xw_plus_b(z, w, b) #[bs, M, N, 1]
    z = tf.reshape(z, [batch_size, passage_len, quetion_len]) #[bs, M, N]
    z = tf.nn.softmax(z) #[bs, M, N]
    def single_instance (x):
        z1 = x[0]
        q1 = x[1]
        return tf.matmul(z1, q1)
    elems = (z, question_rep)
    return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]

def linear_p_bias(passage_rep, question_rep, w, b):
    #b [d, 1]
    p_shape = tf.shape(passage_rep)
    batch_size = p_shape[0]
    passage_len = p_shape[1]
    dim = p_shape[2]
    p_shape = tf.shape(question_rep)
    quetion_len = p_shape[1]

    p_bias = tf.reshape(passage_rep, [-1, dim])
    p_bias = tf.matmul(p_bias, b) #[-1, 1]
    p_bias = tf.reshape(p_bias, [batch_size, passage_len]) #[bs, M]

    p = tf.expand_dims(passage_rep, 2)  # [bs, M, 1, d]
    q = tf.expand_dims(question_rep, 1)  # [bs, 1, N, d]
    z = tf.multiply(p, q)  # [bs, M, N, d]
    z = tf.reshape(z, [-1, dim])
    z = tf.matmul(z, w)  # [bs, M, N, 1]
    z = tf.reshape(z, [batch_size, passage_len, quetion_len])  # [bs, M, N]

    p_bias = tf.expand_dims(p_bias, -1) #[bs, M, 1]
    z = tf.add(p_bias, z)
    z = tf.nn.softmax(z)  # [bs, M, N]
    def single_instance (x):
        z1 = x[0]
        q1 = x[1]
        return tf.matmul(z1, q1)
    elems = (z, question_rep)
    return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]

# def my_att_fw_bw(passage_rep, question_rep,w1, w2):
#     def single_instance (x):
#         q = x[0] #[N,d]
#         p = x[1] #[M,d]
#         z = my_att_base(p, q, w1)
#         p_r = tf.reverse(passage_rep, axis=-1)
#         z = z + my_att_base(p_r, q, w2)
#         z = tf.nn.softmax(z)  # [M,N]
#         return tf.matmul(z, q) #[M,d]
#     elems = (question_rep, passage_rep)
#     return tf.map_fn(single_instance, elems, dtype=tf.float32) #[bs, M, d]

def multi_bilinear_att (passage_rep, question_rep,num_att_type,input_dim , is_shared_attetention, num_call
                        ,with_bilinear_att , overlap, question_mask ,clip_attention, scope = None):
    scope_name = 'bi_att_layer'
    if scope is not None: scope_name = scope
    h_rep_list = []
    for i in range(num_att_type):
        if is_shared_attetention == True:
            cur_scope_name = scope_name + "-{}".format(i)
        else:
            cur_scope_name ="-{}".format(num_call) + scope_name + "-{}".format(i)
        with tf.variable_scope(cur_scope_name, reuse=is_shared_attetention):
            if with_bilinear_att == 'bilinear':
                w = tf.get_variable('bilinear_w', [input_dim, input_dim], dtype=tf.float32)
                b = tf.get_variable('bilinear_b', [input_dim], dtype = tf.float32)
                h_rep_list.append(bilinear_att(passage_rep, question_rep, w, b))
            elif with_bilinear_att == 'linear':
                w= tf.get_variable('linear_w', [input_dim, 1], dtype = tf.float32)
                b = tf.get_variable('linear_b', [1], dtype = tf.float32)
                h_rep_list.append(linear(passage_rep, question_rep, w, b))
            elif with_bilinear_att == 'linear_p_bias':
                w = tf.get_variable('linear_w', [input_dim, 1], dtype=tf.float32)
                b = tf.get_variable('linear_b', [input_dim, 1], dtype=tf.float32)
                h_rep_list.append(linear_p_bias(passage_rep, question_rep, w, b))
            elif with_bilinear_att =='dot_product':
                zz, ans = dot_product(passage_rep, question_rep, input_dim, overlap, question_mask,clip_attention)
                h_rep_list.append(ans)
    return zz, h_rep_list

def sub (x, y):
    in_sub = tf.subtract(x,y)
    in_mul = tf.abs(in_sub)
    return in_mul


def cal_wxb(in_val, scope, output_dim, input_dim, activation = 'relu'):
    #in_val : [bs, M, d]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_val = tf.reshape(in_val, [batch_size*passage_len, input_dim])
    with tf.variable_scope(scope):
        w = tf.get_variable('sim_w', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('sim_b', [output_dim], dtype = tf.float32)
        if activation == 'relu':
            outputs = tf.nn.relu(tf.nn.xw_plus_b(in_val, w, b))
        elif activation == 'tanh':
            outputs = tf.nn.tanh(tf.nn.xw_plus_b(in_val, w, b))
        else:
            outputs = tf.nn.xw_plus_b(in_val, w, b)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_dim])
    return outputs # [bs, M, MP]

def sim_w_con(h_rep, passage_rep, mp_dim, scope, input_dim,activation):
    in_val = tf.concat([h_rep, passage_rep], 2) #[bs, M, 2d]
    return cal_wxb(in_val, scope, mp_dim, 2*input_dim,activation)

def sim_w_mul(h_rep, passage_rep, mp_dim, scope, input_dim,activation):
    in_val = tf.multiply(h_rep, passage_rep) #[bs, M, d]
    #sign = tf.sign(in_val)
    #in_val = tf.abs(in_val)
    #in_val = tf.sqrt(in_val + eps)

    #in_val = tf.multiply(in_val, in_val)

    #in_val = tf.multiply(in_val, sign)
    return cal_wxb(in_val, scope, mp_dim, input_dim,activation)

def sim_w_sub(h_rep, passage_rep, mp_dim, scope, input_dim,activation):
    in_val = sub(h_rep, passage_rep)
    return cal_wxb(in_val, scope, mp_dim, input_dim,activation)

def sim_w_sub_mul(h_rep, passage_rep, mp_dim, scope, input_dim,activation):
    in_mul = tf.multiply(h_rep, passage_rep)
    in_mul = cal_wxb(in_mul, scope, mp_dim, input_dim,activation)
    in_sub = sub(h_rep, passage_rep)
    in_sub = cal_wxb(in_sub, scope + 'sub', mp_dim//2, input_dim,activation)
    in_val = tf.concat([in_mul, in_sub], 2) #[bs, M, 2d]
    return in_val
    #return cal_wxb(in_val, scope, mp_dim, 2*input_dim,activation)

def sim_mul(h_rep, passage_rep, mp_dim, scope, input_dim):
    #[bs, M, d]
    in_mul = tf.multiply(h_rep, passage_rep)
    input_shape = tf.shape(in_mul)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_mul = tf.reshape(in_mul, [batch_size , passage_len, mp_dim, -1])
    return tf.reduce_mean(in_mul, 3) #[bs, M, mp]

def sim_sub(h_rep, passage_rep, mp_dim, scope, input_dim):
    in_sub = tf.subtract(h_rep, passage_rep)
    in_mul = tf.abs(in_sub)#tf.multiply(in_sub, in_sub)
    input_shape = tf.shape(in_mul)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    in_mul = tf.reshape(in_mul, [batch_size , passage_len, mp_dim, -1])
    return tf.reduce_mean(in_mul, 3) #[bs, M, mp]


def sim_layer (h_rep, passage_rep, mp_dim, scope, sim_type, input_dim,activation):
    if sim_type == 'w_con':
        return sim_w_con(h_rep, passage_rep, mp_dim, scope, input_dim,activation)
    elif sim_type == 'w_mul':
        return sim_w_mul(h_rep, passage_rep, mp_dim, scope, input_dim,activation)
    elif sim_type == 'w_sub':
        return sim_w_sub(h_rep, passage_rep, mp_dim, scope,input_dim,activation)
    elif sim_type == 'w_sub_mul':
        return sim_w_sub_mul(h_rep, passage_rep, mp_dim, scope,input_dim,activation)
    # elif sim_type == 'w_cos':
    #     w_cos = tf.get_variable("w_cos_weight", [mp_dim, input_dim], dtype= tf.float32)
    #     return cal_attentive_matching(passage_rep, h_rep, w_cos)
    elif sim_type == 'mul':
        return sim_mul(h_rep, passage_rep, mp_dim, scope,input_dim)
    elif sim_type == 'sub':
        return sim_sub(h_rep, passage_rep, mp_dim, scope, input_dim)
    else:
        print ("there is no true sim type")
        return None

def multi_sim_layer (h_rep_list, passage_rep, mp_dim, sim_type_list, input_dim ,scope, activation):
    #scope_name = 'sim_layer'
    outputs = []
    #if scope is not None: scope_name = scope
    for i in range (len(sim_type_list)):
        cur_scope_name = scope + "-{}".format(i)
        outputs.append(sim_layer(h_rep_list[i], passage_rep, mp_dim,
                                 cur_scope_name, sim_type_list[i], input_dim,activation))

    outputs = tf.concat(outputs,2) #[bs, M, num_sim*MP]
    return outputs


def match_bilinear_sim (passage_rep, question_rep, mp_dim, input_dim,
                        type1, type2, type3, is_shared_attetention, num_call, with_bilinear_att
                        , with_match_highway, overlap,question_mask ,clip_attention):
    # passage_rep  [bs, M, d]
    # question_rep [bs, N, d]
    # type means sim_func_type
    sim_type_list = []
    if (type1 is not None): sim_type_list.append(type1)
    if (type2 is not None): sim_type_list.append(type2)
    if (type3 is not None): sim_type_list.append(type3)
    alph, h_rep_list = multi_bilinear_att(passage_rep, question_rep, len(sim_type_list),input_dim, is_shared_attetention, num_call, with_bilinear_att, overlap=overlap,
                                          question_mask=question_mask,clip_attention=clip_attention)
    if with_match_highway == False:
        ans =  multi_sim_layer(h_rep_list, passage_rep, mp_dim, sim_type_list, input_dim, scope=str(num_call), activation='relu')
    else:
        ans = multi_sim_layer(h_rep_list, passage_rep, mp_dim, sim_type_list, input_dim, scope=str(num_call), activation='None')
    l = [ans] #[bs,M,d]

    # p = tf.expand_dims(passage_rep, 2) #[bs, M, 1, d]
    # q = tf.expand_dims(question_rep, 1) #[bs, 1, N, d]
    # z = tf.multiply(p, q) #[bs, M, N, d]
    # o1 = tf.expand_dims(overlap, -1)  # [bs,M, N,1]
    # o1 = tf.multiply(o1, z)  # [bs, M,N,d]
    # o1 = cal_wxb(o1, scope='overlap', output_dim=1, input_dim=input_dim, activation='relu')  # [bs,M,N,1]

    #alph = tf.multiply(alph, overlap) #[bs, M, N]
    #alph = tf.reduce_sum(alph, 2, keep_dims=True) #[bs, M, 1] alpha_i is sum of the weights of q_j for p_i where q_j=p_i
    o = tf.reduce_max(overlap, -1, keep_dims=True) #[bs, M, 1]
    p = tf.multiply(passage_rep, o)
    p = cal_wxb(p,'mp_overlap' + str(num_call), 5, input_dim,activation='relu') #[bs, M, 10]

    attention_weights = alph

    alph = p
    #alph = tf.multiply(alph, p) #[bs, M, 10]
        #alph = tf.reduce_sum(alph, 2, keep_dims=True) #[bs, M, 5]
    l.append(alph)
    return tf.concat(l, 2), attention_weights #[bs, M, d+10]




def self_attention(passage_context_representation_fw, pasage_context_representation_bw,mask,input_dim):
    pasage_context_representation_bw = tf.multiply(pasage_context_representation_bw,
                                                     tf.expand_dims(mask, -1))
    passage_context_representation_fw = tf.multiply(passage_context_representation_fw,
                                                     tf.expand_dims(mask, -1))
    passage_rep = tf.concat([passage_context_representation_fw,pasage_context_representation_bw], 2)
    shrinking_factor = 2
    mm = cal_wxb(passage_rep, scope='ag_att_1',
                            output_dim=input_dim/shrinking_factor, input_dim=input_dim,activation='tanh')  # [bs, M, d/2]
    mm = cal_wxb(mm, scope='ag_att_2',
                 output_dim=1, input_dim=input_dim/shrinking_factor, activation='None')  # [bs, M, 1]
    agg_shape = tf.shape(mm)
    batch_size = agg_shape[0]
    passage_len = agg_shape[1]
    mm = tf.reshape(mm, [batch_size, passage_len])  # [bs, M]
    mm = tf.nn.softmax(mm)
    mm = tf.expand_dims(mm, axis=-1)  # [bs, M, 1]
    mm = tf.multiply(mm, passage_rep)  # [bs, M, d]
    mm = tf.reduce_mean(mm, axis=1)  # [bs,d]
    return mm


def highway_layer(in_val, input_size, scope=None, output_size=-1, with_highway = False):
    if output_size == -1:
        output_size = input_size
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [input_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        if with_highway == False:
            outputs = tf.multiply(trans, gate)#tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y") aslan dige nemishe chon ouput_size != input_size
        else:
            outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

# def cal_max_question_representation(question_representation, cosine_matrix):
#     # question_representation: [batch_size, question_len, dim]
#     # cosine_matrix: [batch_size, passage_len, question_len]
#     question_index = tf.arg_max(cosine_matrix, 2) # [batch_size, passage_len]
#     def singel_instance(x):
#         q = x[0]
#         c = x[1]
#         return tf.gather(q, c)
#     elems = (question_representation, question_index)
#     return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, dim]
#
# def cal_linear_decomposition_representation(passage_representation, passage_lengths, cosine_matrix,is_training,
#                                             lex_decompsition_dim, dropout_rate):
#     # passage_representation: [batch_size, passage_len, dim]
#     # cosine_matrix: [batch_size, passage_len, question_len]
#     passage_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, passage_len]
#     similar_weights = tf.expand_dims(passage_similarity, -1) # [batch_size, passage_len, 1]
#     dissimilar_weights = tf.subtract(1.0, similar_weights)
#     similar_component = tf.mul(passage_representation, similar_weights)
#     dissimilar_component = tf.mul(passage_representation, dissimilar_weights)
#     all_component = tf.concat(2, [similar_component, dissimilar_component])
#     if lex_decompsition_dim==-1:
#         return all_component
#     with tf.variable_scope('lex_decomposition'):
#         lex_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
#         lex_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
#         if is_training:
#             lex_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
#             lex_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
#         lex_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_fw])
#         lex_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_bw])
#
#         (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
#                     lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=passage_lengths)
#
#         lex_features = tf.concat(2, [lex_features_fw, lex_features_bw])
#     return lex_features

def match_passage_with_question(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                                with_bilinear_att = 's', type1 = None, type2 = None, type3= None,
                                is_shared_attetention = False, unstack_cnn = True, num_call = 1, with_match_highway = True, overlap=None
                                ,clip_attention = True, with_input_highway = False):

    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        # fw_question_full_rep = question_context_representation_fw[:,-1,:]
        # bw_question_full_rep = question_context_representation_bw[:,0,:]
        if with_input_highway == False:
            question_context_representation_bw = tf.multiply(question_context_representation_bw, tf.expand_dims(question_mask,-1))
            passage_context_representation_bw = tf.multiply(passage_context_representation_bw, tf.expand_dims(mask, -1))
        passage_context_representation_fw = tf.multiply(passage_context_representation_fw, tf.expand_dims(mask,-1))
        question_context_representation_fw = tf.multiply(question_context_representation_fw,
                                                         tf.expand_dims(question_mask, -1))

        # forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        # forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)
        #
        # backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        # backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)
        if with_input_highway == False:
            question_context_representation_fw_bw = tf.concat([question_context_representation_fw,
                                                           question_context_representation_bw], 2)
            passage_context_representation_fw_bw = tf.concat([passage_context_representation_fw,
                                                          passage_context_representation_bw], 2)
            outputs, attention_weights = match_bilinear_sim(passage_context_representation_fw_bw, question_context_representation_fw_bw,
                           MP_dim,context_lstm_dim*2,type1, type2, type3, is_shared_attetention, num_call, with_bilinear_att, with_match_highway
                                     ,overlap,question_mask ,clip_attention)
        else:
            outputs, attention_weights = match_bilinear_sim(passage_context_representation_fw, question_context_representation_fw,
                                         MP_dim, context_lstm_dim, type1, type2, type3, is_shared_attetention,
                                         num_call, with_bilinear_att, with_match_highway
                                         , overlap, question_mask,clip_attention)
        all_question_aware_representatins.append(outputs)
        if type1 is not None: dim += MP_dim
        if type2 is not None: dim += MP_dim
        if type3 is not None: dim += MP_dim
        dim += 10 #overlap
        # if MP_dim > 0:
        #     if with_full_match:
        #         # forward Full-Matching: passage_context_representation_fw vs question_context_representation_fw[-1]
        #         fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         fw_full_match_rep = cal_full_matching(passage_context_representation_fw, fw_question_full_rep, fw_full_decomp_params)
        #         all_question_aware_representatins.append(fw_full_match_rep)
        #         dim += MP_dim
        #
        #         # backward Full-Matching: passage_context_representation_bw vs question_context_representation_bw[0]
        #         bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         bw_full_match_rep = cal_full_matching(passage_context_representation_bw, bw_question_full_rep, bw_full_decomp_params)
        #         all_question_aware_representatins.append(bw_full_match_rep)
        #         dim += MP_dim
        #
        #     if with_maxpool_match:
        #         # forward Maxpooling-Matching
        #         fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         fw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_fw, question_context_representation_fw, fw_maxpooling_decomp_params)
        #         all_question_aware_representatins.append(fw_maxpooling_rep)
        #         dim += 2*MP_dim
        #         # backward Maxpooling-Matching
        #         bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         bw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_bw, question_context_representation_bw, bw_maxpooling_decomp_params)
        #         all_question_aware_representatins.append(bw_maxpooling_rep)
        #         dim += 2*MP_dim
        #
        #     if with_attentive_match:
        #         # forward attentive-matching
        #         # forward weighted question representation: [batch_size, question_len, passage_len] [batch_size, question_len, context_lstm_dim]
        #         att_question_fw_contexts = cal_cosine_weighted_question_representation(question_context_representation_fw, forward_relevancy_matrix)
        #         fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         fw_attentive_rep = cal_attentive_matching(passage_context_representation_fw, att_question_fw_contexts, fw_attentive_decomp_params)
        #         all_question_aware_representatins.append(fw_attentive_rep)
        #         dim += MP_dim
        #
        #         # backward attentive-matching
        #         # backward weighted question representation
        #         att_question_bw_contexts = cal_cosine_weighted_question_representation(question_context_representation_bw, backward_relevancy_matrix)
        #         bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         bw_attentive_rep = cal_attentive_matching(passage_context_representation_bw, att_question_bw_contexts, bw_attentive_decomp_params)
        #         all_question_aware_representatins.append(bw_attentive_rep)
        #         dim += MP_dim
        #
        #     if with_max_attentive_match:
        #         # forward max attentive-matching
        #         max_att_fw = cal_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
        #         fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         fw_max_attentive_rep = cal_attentive_matching(passage_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
        #         all_question_aware_representatins.append(fw_max_attentive_rep)
        #         dim += MP_dim
        #
        #         # backward max attentive-matching
        #         max_att_bw = cal_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
        #         bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
        #         bw_max_attentive_rep = cal_attentive_matching(passage_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
        #         all_question_aware_representatins.append(bw_max_attentive_rep)
        #         dim += MP_dim
        #
        # #all_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        # #all_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        # #all_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        # #all_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        # #dim += 4
    return (all_question_aware_representatins, dim, attention_weights)
# def unidirectional_matching(in_question_repres, in_passage_repres,question_lengths, passage_lengths,
#                             question_mask, mask, MP_dim, input_dim, with_filter_layer, context_layer_num,
#                             context_lstm_dim,is_training,dropout_rate,with_match_highway,aggregation_layer_num,
#                             aggregation_lstm_dim,highway_layer_num,with_aggregation_highway,with_lex_decomposition, lex_decompsition_dim,
#                             with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):
#     # ======Filter layer======
#     cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres)
#     cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
#     raw_in_passage_repres = in_passage_repres
#     if with_filter_layer:
#         relevancy_matrix = cosine_matrix # [batch_size, passage_len, question_len]
#         relevancy_degrees = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size, passage_len]
#         relevancy_degrees = tf.expand_dims(relevancy_degrees,axis=-1) # [batch_size, passage_len, 'x']
#         in_passage_repres = tf.mul(in_passage_repres, relevancy_degrees)
#
#     # =======Context Representation Layer & Multi-Perspective matching layer=====
#     all_question_aware_representatins = []
#     # max and mean pooling at word level
#     all_question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True))
#     all_question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))
#     question_aware_dim = 2
#
#     if MP_dim>0:
#         if with_max_attentive_match:
#             # max_att word level
#             max_att = cal_max_question_representation(in_question_repres, cosine_matrix)
#             max_att_decomp_params = tf.get_variable("max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
#             max_attentive_rep = cal_attentive_matching(raw_in_passage_repres, max_att, max_att_decomp_params)
#             all_question_aware_representatins.append(max_attentive_rep)
#             question_aware_dim += MP_dim
#
#     # lex decomposition
#     if with_lex_decomposition:
#         lex_decomposition = cal_linear_decomposition_representation(raw_in_passage_repres, passage_lengths, cosine_matrix,is_training,
#                                             lex_decompsition_dim, dropout_rate)
#         all_question_aware_representatins.append(lex_decomposition)
#         if lex_decompsition_dim== -1: question_aware_dim += 2 * input_dim
#         else: question_aware_dim += 2* lex_decompsition_dim
#
#     with tf.variable_scope('context_MP_matching'):
#         for i in xrange(context_layer_num):
#             with tf.variable_scope('layer-{}'.format(i)):
#                 with tf.variable_scope('context_represent'):
#                     # parameters
#                     context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
#                     context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
#                     if is_training:
#                         context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
#                         context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
#                     context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
#                     context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])
#
#                     # question representation
#                     (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                         context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32,
#                                         sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
#                     in_question_repres = tf.concat(2, [question_context_representation_fw, question_context_representation_bw])
#
#                     # passage representation
#                     tf.get_variable_scope().reuse_variables()
#                     (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                         context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32,
#                                         sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
#                     in_passage_repres = tf.concat(2, [passage_context_representation_fw, passage_context_representation_bw])
#
#                 # Multi-perspective matching
#                 with tf.variable_scope('MP_matching'):
#                     (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw,
#                                 passage_context_representation_bw, mask,
#                                 question_context_representation_fw, question_context_representation_bw,question_mask,
#                                 MP_dim, context_lstm_dim, scope=None,
#                                 with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
#                                 with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
#                     all_question_aware_representatins.extend(matching_vectors)
#                     question_aware_dim += matching_dim
#
#     all_question_aware_representatins = tf.concat(2, all_question_aware_representatins) # [batch_size, passage_len, dim]
#
#     if is_training:
#         all_question_aware_representatins = tf.nn.dropout(all_question_aware_representatins, (1 - dropout_rate))
#     else:
#         all_question_aware_representatins = tf.mul(all_question_aware_representatins, (1 - dropout_rate))
#
#     # ======Highway layer======
#     if with_match_highway:
#         with tf.variable_scope("matching_highway"):
#             all_question_aware_representatins = multi_highway_layer(all_question_aware_representatins, question_aware_dim,highway_layer_num)
#
#     #========Aggregation Layer======
#     aggregation_representation = []
#     aggregation_dim = 0
#     aggregation_input = all_question_aware_representatins
#     with tf.variable_scope('aggregation_layer'):
#         for i in xrange(aggregation_layer_num):
#             with tf.variable_scope('layer-{}'.format(i)):
#                 aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
#                 aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
#                 if is_training:
#                     aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
#                     aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
#                 aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
#                 aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])
#
#                 cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
#                         aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input,
#                         dtype=tf.float32, sequence_length=passage_lengths)
#
#                 fw_rep = cur_aggregation_representation[0][:,-1,:]
#                 bw_rep = cur_aggregation_representation[1][:,0,:]
#                 aggregation_representation.append(fw_rep)
#                 aggregation_representation.append(bw_rep)
#                 aggregation_dim += 2* aggregation_lstm_dim
#                 aggregation_input = tf.concat(2, cur_aggregation_representation)# [batch_size, passage_len, 2*aggregation_lstm_dim]
#
#     #
#     aggregation_representation = tf.concat(1, aggregation_representation) # [batch_size, aggregation_dim]
#
#     # ======Highway layer======
#     if with_aggregation_highway:
#         with tf.variable_scope("aggregation_highway"):
#             agg_shape = tf.shape(aggregation_representation)
#             batch_size = agg_shape[0]
#             aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
#             aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
#             aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
#
#     return (aggregation_representation, aggregation_dim)
#
# def bilateral_match_func1(in_question_repres, in_passage_repres,
#                         question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim,
#                         with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
#                         with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
#                         with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
#                         with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
#                         with_left_match=True, with_right_match=True):
#     init_scale = 0.01
#     initializer = tf.random_uniform_initializer(-init_scale, init_scale)
#     match_representation = []
#     match_dim = 0
#
#     reuse_match_params = None
#     if with_left_match:
#         reuse_match_params = True
#         with tf.name_scope("match_passsage"):
#             with tf.variable_scope("MP-Match", reuse=None, initializer=initializer):
#                 (passage_match_representation, passage_match_dim) = unidirectional_matching(in_question_repres, in_passage_repres,
#                             question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim,
#                             with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
#                             with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
#                             with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
#                             with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
#                             with_attentive_match=with_attentive_match,
#                             with_max_attentive_match=with_max_attentive_match)
#                 match_representation.append(passage_match_representation)
#                 match_dim += passage_match_dim
#     if with_right_match:
#         with tf.name_scope("match_question"):
#             with tf.variable_scope("MP-Match", reuse=reuse_match_params, initializer=initializer):
#                 (question_match_representation, question_match_dim) = unidirectional_matching(in_passage_repres, in_question_repres,
#                             passage_lengths, question_lengths, mask, question_mask, MP_dim, input_dim,
#                             with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
#                             with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
#                             with_aggregation_highway, with_lex_decomposition,lex_decompsition_dim,
#                             with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
#                             with_attentive_match=with_attentive_match,
#                             with_max_attentive_match=with_max_attentive_match)
#                 match_representation.append(question_match_representation)
#                 match_dim += question_match_dim
#     match_representation = tf.concat(1, match_representation)
#     return (match_representation, match_dim)
#


def bilateral_match_func2(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True,
                          with_bilinear_att = 's', type1 = None, type2 = None, type3 = None, with_aggregation_attention = True,
                          is_shared_attetention = True, is_aggregation_lstm = True, max_window_size = 3,
                          context_lstm_dropout = True, is_aggregation_siamese = False, unstack_cnn = True, with_input_highway=False, with_context_self_attention=False, overlap = None,
                          mean_max = True, clip_attention = True):

    # ====word level matching======
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    quesstion_self_att = 0
    passage_self_att = 0
    # cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    # cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    # cosine_matrix_transpose = tf.transpose(cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]
    # max and mean pooling at word level
    #question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True)) # [batch_size, passage_length, 1]
    #question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))# [batch_size, passage_length, 1]
    #question_aware_dim += 2
    #passage_aware_representatins.append(tf.reduce_max(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    #passage_aware_representatins.append(tf.reduce_mean(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    #passage_aware_dim += 2

    # if MP_dim>0: #0 boode
    #     if with_max_attentive_match:
    #         # max_att word level
    #         qa_max_att = cal_max_question_representation(in_question_repres, cosine_matrix)# [batch_size, passage_len, dim]
    #         qa_max_att_decomp_params = tf.get_variable("qa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
    #         qa_max_attentive_rep = cal_attentive_matching(in_passage_repres, qa_max_att, qa_max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
    #         question_aware_representatins.append(qa_max_attentive_rep)
    #         question_aware_dim += MP_dim
    #
    #         pa_max_att = cal_max_question_representation(in_passage_repres, cosine_matrix_transpose)# [batch_size, question_len, dim]
    #         pa_max_att_decomp_params = tf.get_variable("pa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
    #         pa_max_attentive_rep = cal_attentive_matching(in_question_repres, pa_max_att, pa_max_att_decomp_params)# [batch_size, question_len, decompse_dim]
    #         passage_aware_representatins.append(pa_max_attentive_rep)
    #         passage_aware_dim += MP_dim
    with tf.variable_scope('context_MP_matching'):
        for i in range(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                if with_input_highway == False:
                    with tf.variable_scope('context_represent'):
                        # parameters
                        #context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                        #context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                        context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                        context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                        if is_training and context_lstm_dropout == True:
                        #     context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        #     context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                        # context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                        # context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])
                            context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw,
                                                                             output_keep_prob=(1 - dropout_rate))
                            context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw,
                                                                            output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                        context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])

                        # question representation
                        (question_context_representation_fw, question_context_representation_bw), _ = rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32,
                                            sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                        in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)
                        # passage representation
                        tf.get_variable_scope().reuse_variables()
                        (passage_context_representation_fw, passage_context_representation_bw), _ = rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32,
                                            sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                        in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)
                    if i == 0:
                        with tf.variable_scope ("context_self"):
                            passage_self_att = self_attention(passage_context_representation_fw=passage_context_representation_fw,
                                       pasage_context_representation_bw=passage_context_representation_bw,
                                       mask=mask, input_dim=context_lstm_dim * 2)
                            tf.get_variable_scope().reuse_variables()
                            question_self_att = self_attention(passage_context_representation_fw=question_context_representation_fw,
                                   pasage_context_representation_bw=question_context_representation_bw,
                                   mask=question_mask, input_dim=context_lstm_dim * 2)
                else:
                    passage_context_representation_fw = in_passage_repres
                    passage_context_representation_bw = None
                    question_context_representation_fw = in_question_repres
                    question_context_representation_bw = None
                # else:
                #     passage_context_representation_fw = in_passage_repres
                #     passage_context_representation_bw = in_passage_repres
                #     question_context_representation_fw = in_question_repres
                #     question_context_representation_bw = in_question_repres

                # Multi-perspective matching
                with tf.variable_scope('MP_matching'):
                    (matching_vectors, matching_dim, attention_weights) = match_passage_with_question(passage_context_representation_fw,
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                    with_bilinear_att=with_bilinear_att, type1=type1, type2=type2, type3=type3
                                                                                   ,is_shared_attetention = False,
                                                                                   unstack_cnn= unstack_cnn, num_call = 1,
                                                                                   with_match_highway=with_match_highway,
                                                                                   overlap=tf.transpose(overlap,perm=[0,2,1])
                                                                                   , clip_attention=clip_attention,
                                                                                   with_input_highway=with_input_highway)
                    question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
                #right_scope = 'right_MP_matching'
                #if is_shared_attetention == True:
                #    right_scope = 'left_MP_matching'
                #with tf.variable_scope('MP_matching', reuse=is_shared_attetention):
                    (matching_vectors, matching_dim, attention_weights) = match_passage_with_question(question_context_representation_fw,
                                question_context_representation_bw, question_mask,
                                passage_context_representation_fw, passage_context_representation_bw,mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                    with_bilinear_att=with_bilinear_att, type1=type1, type2=type2, type3 = type3
                                                                                   ,is_shared_attetention = False, unstack_cnn=unstack_cnn,
                                                                                   num_call = 2, with_match_highway=with_match_highway, overlap=overlap
                                                                                   , clip_attention=clip_attention,
                                                                                   with_input_highway=with_input_highway)
                    passage_aware_representatins.extend(matching_vectors)
                    passage_aware_dim += matching_dim
        


    question_aware_representatins = tf.concat(question_aware_representatins, 2) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(passage_aware_representatins, 2) # [batch_size, question_len, question_aware_dim]
    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - dropout_rate))
    else:
        question_aware_representatins = tf.multiply(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.multiply(passage_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    '''
    if with_mean_aggregation:
        aggregation_representation.append(tf.reduce_mean(question_aware_representatins, axis=1))
        aggregation_dim += question_aware_dim
        aggregation_representation.append(tf.reduce_mean(passage_aware_representatins, axis=1))
        aggregation_dim += passage_aware_dim
    #'''

    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        if is_aggregation_lstm == True:
            for i in range(aggregation_layer_num): # support multiple aggregation layer
                my_scope = 'left_layer-{}'.format(i)
                my_reuse = True
                with tf.variable_scope(my_scope):
                    aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    if is_training:
                        aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                    aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                    (passage_aggregation_representation_fw,passage_aggregation_representation_bw) , _ = rnn.bidirectional_dynamic_rnn(
                            aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, qa_aggregation_input,
                            dtype=tf.float32, sequence_length=passage_lengths)
                    qa_aggregation_input = tf.concat([passage_aggregation_representation_fw,passage_aggregation_representation_bw], 2)# [batch_size, passage_len, 2*aggregation_lstm_dim]
                    if with_aggregation_attention == False:
                        if mean_max == False:
                            fw_rep = passage_aggregation_representation_fw[:,-1,:]
                            bw_rep = passage_aggregation_representation_bw[:,0,:]
                            aggregation_representation.append(fw_rep)
                            aggregation_representation.append(bw_rep)
                            aggregation_dim += 2 * aggregation_lstm_dim
                        else:
                            aggregation_representation.append(get_mean_max (passage_aggregation_representation_fw,passage_lengths,mask))
                            aggregation_dim += 2*aggregation_lstm_dim # max, mean
                    else:
                        aggregation_representation.append(
                            self_attention(passage_aggregation_representation_fw,passage_aggregation_representation_bw, mask,
                                                  aggregation_lstm_dim * 2))
                        aggregation_dim += 2 * aggregation_lstm_dim

                if is_aggregation_siamese == False:
                    my_scope = 'right_layer-{}'.format(i)
                    my_reuse = False
                with tf.variable_scope(my_scope, reuse=my_reuse):
                    aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    if is_training:
                        aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                    aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                    (question_aggregation_representation_fw, question_aggregation_representation_bw), _ = rnn.bidirectional_dynamic_rnn(
                            aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, pa_aggregation_input,
                            dtype=tf.float32, sequence_length=question_lengths)
                    pa_aggregation_input = tf.concat([question_aggregation_representation_fw,question_aggregation_representation_bw],2)# [batch_size, passage_len, 2*aggregation_lstm_dim]

                    if with_aggregation_attention == False:
                        if mean_max == False:
                            fw_rep = question_aggregation_representation_fw[:,-1,:]
                            bw_rep = question_aggregation_representation_bw[:,0,:]
                            aggregation_representation.append(fw_rep)
                            aggregation_representation.append(bw_rep)
                            aggregation_dim += 2* aggregation_lstm_dim
                        else:
                            aggregation_representation.append(get_mean_max (question_aggregation_representation_fw,question_lengths,question_mask))
                            aggregation_dim += 2*aggregation_lstm_dim # max, mean
                    else:
                        aggregation_representation.append(
                            self_attention(question_aggregation_representation_fw,question_aggregation_representation_bw
                                                  ,question_mask,aggregation_lstm_dim * 2))
                        aggregation_dim += 2 * aggregation_lstm_dim
            if aggregation_layer_num == 2:
                aggregation_representation = aggregation_representation [len (aggregation_representation) // 2:]
                aggregation_dim = aggregation_dim // 2
        else: #CNN
            sim_len = 0
            if type1 != None:
                sim_len += 1
            if type2 != None:
                sim_len += 1
            if type3 != None:
                sim_len += 1

            my_scope = 'left_cnn_agg'
            my_reuse = True
            with tf.variable_scope ('left_cnn_agg'):
                conv_out, agg_dim = conv_aggregate(pa_aggregation_input, aggregation_lstm_dim, MP_dim, sim_len,is_training,dropout_rate
                                               ,max_window_size, context_layer_num,unstack_cnn)
                aggregation_representation.append(conv_out)
                aggregation_dim += agg_dim
            if is_aggregation_siamese == False:
                my_scope = 'right_cnn_agg'
                my_reuse = False
            with tf.variable_scope(my_scope, reuse=my_reuse):
                conv_out, agg_dim = conv_aggregate(qa_aggregation_input, aggregation_lstm_dim, MP_dim, sim_len, is_training, dropout_rate
                                                   ,max_window_size, context_layer_num,unstack_cnn)
                aggregation_representation.append(conv_out)
                aggregation_dim += agg_dim
    #

    if with_context_self_attention == True:
        aggregation_representation.append(tf.subtract(question_self_att , passage_self_att))
        aggregation_representation.append(tf.multiply(question_self_att , passage_self_att))
        aggregation_dim += 4*context_lstm_dim
    aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])


    return (aggregation_representation, aggregation_dim, attention_weights)


def get_mean_max (passage_rep, passage_length, mask):
    #pas_rep : [bs, M, d]
    #pas_len: [bs]
    passage_rep = tf.multiply(passage_rep,
                tf.expand_dims(mask, -1))
    max_pool = tf.reduce_max(passage_rep,1) #[bs, d]
    passage_length1 = tf.expand_dims(passage_length,-1) #[bs,1]
    passage_length1 = tf.to_float(passage_length1)
    mean_pool = tf.divide(tf.reduce_sum(passage_rep, 1), passage_length1) #[bs,d]
    ans = tf.concat([mean_pool, max_pool], 1) #[bs, 2d]
    return ans



def conv_pooling (passage_rep, window_size, mp_dim, filter_count, scope):
    # passage_rep : [bs, M, mp]
    filter_width = window_size
    in_channels = mp_dim
    out_channels = filter_count
    with tf.variable_scope(scope):
        w = tf.get_variable("filter_w", [filter_width, in_channels, out_channels], dtype=tf.float32)
        conv = tf.nn.relu(tf.nn.conv1d(value=passage_rep, filters=w, stride=1, padding='SAME')) #[bs, M, out_channels]
        conv = tf.reduce_max(conv, axis = 1) #[bs, out_channels]
        aggregation_dim = out_channels
        return conv, aggregation_dim


def conv_aggregate(qa_aggregation_input, aggregation_lstm_dim, mp_dim, sim_len, is_training, dropout_rate, max_window_size
                   , c_lstm_layer, unstack_cnn = True):
    qa_shape = tf.shape(qa_aggregation_input)  # [bs, M, MP*sim_len]
    batch_size = qa_shape[0]
    passage_length = qa_shape[1]
    if unstack_cnn == True:
        passage_rep = tf.reshape(qa_aggregation_input, [batch_size, passage_length, mp_dim, sim_len*c_lstm_layer]) #-1: sim_len*c_lstm_layer
        passage_rep = tf.unstack(passage_rep, axis=3) #[[bs, M, MP]]
    else:
        passage_rep = [qa_aggregation_input]
    aggregation_dim = 0
    passage_cnn_output = []
    for filter_size in range (1, max_window_size + 1):
        for i in range(len(passage_rep)):
            cur_scope_name = "-{}-{}".format(filter_size, i)
            if unstack_cnn == True:
                conv_out, dim = conv_pooling(passage_rep[i], window_size=filter_size, mp_dim=mp_dim,
                                         filter_count=aggregation_lstm_dim, scope=cur_scope_name)
            else:
                conv_out, dim = conv_pooling(passage_rep[i], window_size=filter_size, mp_dim=mp_dim*sim_len*c_lstm_layer,
                                             filter_count=aggregation_lstm_dim, scope=cur_scope_name)
            passage_cnn_output.append(conv_out)  # [bs, filter_count]
            aggregation_dim += dim
    passage_cnn_output = tf.concat(passage_cnn_output,1)  # [bs, filter_count*sim_len]

    w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
    b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)

    logits = tf.matmul(passage_cnn_output, w_0) + b_0
    logits = tf.nn.tanh(logits)
    #if is_training:
    #    logits = tf.nn.dropout(logits, (1 - dropout_rate))
    #else:
    #    logits = tf.mul(logits, (1 - dropout_rate))

    return logits, aggregation_dim/2
