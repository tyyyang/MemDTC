# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Visual Tracking via Dynamic Memory Networks", TPAMI, 2019
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import collections
import tensorflow as tf
from memnet.access import MemoryAccessPos, MemoryAccessNeg, AccessStatePos, AccessStateNeg
from feature import get_key_feature
import config
import numpy as np

MemNetState = collections.namedtuple('MemNetState', ('controller_state', 'access_state_pos', 'access_state_neg', 'neg_template', 'neg_idx'))

def attention(input, query, scope=None):

    input_shape = input.get_shape().as_list()
    input_transform = tf.layers.conv2d(input, input_shape[-1], [1, 1], [1, 1], use_bias=False, name='input_layer')
    query_transform = tf.layers.dense(query, input_shape[-1], name='query_layer')
    query_transform = tf.expand_dims(tf.expand_dims(query_transform, 1), 1)
    addition = tf.nn.tanh(input_transform + query_transform, name='addition')
    addition_transform = tf.layers.conv2d(addition, 1, [1, 1], [1, 1], use_bias=False, name='score')
    addition_shape = addition_transform.get_shape().as_list()
    score = tf.nn.softmax(tf.reshape(addition_transform, [addition_shape[0], -1]))

    if int(scope) < config.summary_display_step:
        max_idxes = tf.argmax(score, 1)
        tf.summary.histogram('max_idxes_{}'.format(scope),max_idxes)
        max_value = tf.reduce_max(score, 1)
        tf.summary.histogram('max_value_{}'.format(scope), max_value)

    score = tf.reshape(score, addition_shape)
    return tf.reduce_sum(input*score, [1,2]), score

def extract_neg_template(response, search_feature, num_write):

    response_shape = response.shape
    response_flat = np.reshape(response, [response_shape[0], -1])
    max_score = np.max(response_flat, -1)
    max_pos = np.argmax(response_flat, -1)
    dist = lambda i, j, orgin: np.linalg.norm(np.array([i, j]) - orgin)
    target_loc = np.array([max_pos // response_shape[2], max_pos % response_shape[1]])
    for k in range(response_shape[0]):
        for i in range(response_shape[1]):
            for j in range(response_shape[2]):
                if dist(i, j, target_loc[:, k]) <= config.neg_dist_thre:
                    response_flat[k, i * response_shape[1] + j] = -np.inf

    neg_idxes = np.flip(np.argsort(response_flat, -1), -1)
    neg_templates = np.zeros([response_shape[0], num_write] + config.slot_size, np.float32)
    neg_idx_chosen = np.zeros([response_shape[0], num_write], np.int32)

    for m in range(response_shape[0]):
        for n in range(num_write):
            idx = neg_idxes[m, n]
            if response_flat[m, idx] > config.neg_score_ratio * max_score[m]:
                x = idx // response_shape[2]
                y = idx % response_shape[2]
                # print(x, y)
                side = config.slot_size[0]
                neg_templates[m, n] = search_feature[m, x:x + side, y:y + side, :]
                neg_idx_chosen[m, n] = idx
    return neg_templates, neg_idx_chosen

class MemNet(tf.nn.rnn_cell.RNNCell):

    def __init__(self, hidden_size, memory_size_pos, memory_size_neg, slot_size, is_train):
        super(MemNet, self).__init__()
        # self._controller = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        # if is_train and config.keep_prob < 1:
        #     self._controller = tf.nn.rnn_cell.DropoutWrapper(self._controller,
        #                                                      input_keep_prob=config.keep_prob,
        #                                                      output_keep_prob=config.keep_prob)
        keep_prob = config.keep_prob if is_train else 1.0
        self._controller = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, layer_norm=True, dropout_keep_prob=keep_prob)
        self._memory_access_pos = MemoryAccessPos(memory_size_pos, slot_size, is_train)
        self._memory_access_neg = MemoryAccessNeg(memory_size_neg, slot_size, is_train)
        self._hidden_size = hidden_size
        self._memory_size_pos = memory_size_pos
        self._memory_size_neg = memory_size_neg
        self._slot_size = slot_size
        self._is_train = is_train

    def __call__(self, inputs, prev_state, scope=None):

        prev_controller_state = prev_state.controller_state
        prev_access_state_pos = prev_state.access_state_pos
        prev_access_state_neg = prev_state.access_state_neg

        search_feature = inputs[0]
        memory_for_writing = inputs[1]

        # get lstm controller input
        controller_input = get_key_feature(search_feature, self._is_train, 'search_key')

        attention_input, self.att_score = attention(controller_input, prev_controller_state[1], scope)

        controller_output, controller_state = self._controller(attention_input, prev_controller_state, scope)

        pos_access_inputs = (memory_for_writing, controller_output)
        access_output_pos, access_state_pos = self._memory_access_pos(pos_access_inputs, prev_access_state_pos, scope)
        neg_access_inputs = (prev_state.neg_template, controller_output)
        access_output_neg, access_state_neg = self._memory_access_neg(neg_access_inputs, prev_access_state_neg, scope)


        trans_pos = tf.layers.conv2d(access_output_pos, config.slot_size[2], [1,1], use_bias=False,
                                     name='trans_pos')
        trans_neg = tf.layers.conv2d(access_output_neg, config.slot_size[2], [1,1], name='trans_neg')
        adap_gates = tf.sigmoid(tf.layers.conv2d(tf.nn.tanh(trans_pos + trans_neg), config.slot_size[2],
                                                 config.slot_size[0:2], use_bias=False, name='adap_gates'))
        access_output = access_output_pos - adap_gates * access_output_neg

        if int(scope) < config.summary_display_step:
            tf.summary.histogram('adap_gates/{}'.format(scope), adap_gates)

        batch_size = search_feature.get_shape().as_list()[0]
        response = tf.map_fn(
            lambda inputs: tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], 3), [1, 1, 1, 1],'VALID'),
            elems=[search_feature, access_output],
            dtype=tf.float32,
            parallel_iterations=batch_size)
        response = tf.squeeze(tf.squeeze(response, 1),-1)
        neg_template, neg_idx = tf.py_func(extract_neg_template, [response, search_feature, config.neg_num_write], [tf.float32, tf.int32])
        neg_template.set_shape([batch_size, config.neg_num_write]+config.slot_size)

        return access_output, MemNetState(access_state_pos=access_state_pos, access_state_neg=access_state_neg, controller_state=controller_state,
                                          neg_template=neg_template, neg_idx = neg_idx)

    def initial_state(self, init_feature):

        init_key = tf.squeeze(get_key_feature(init_feature, self._is_train, 'init_memory_key'), [1, 2])
        c_state = tf.layers.dense(init_key, self._hidden_size, activation=tf.nn.tanh, name='c_state')
        h_state = tf.layers.dense(init_key, self._hidden_size, activation=tf.nn.tanh, name='h_state')
        batch_size = init_key.get_shape().as_list()[0]
        controller_state = tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state)

        write_weights = tf.one_hot([0]*batch_size, self._memory_size_pos, axis=-1, dtype=tf.float32)
        read_weight = tf.zeros([batch_size, self._memory_size_pos], tf.float32)
        control_factors = tf.one_hot([2]*batch_size, 3, axis=-1, dtype=tf.float32)
        write_decay = tf.zeros([batch_size, 1], tf.float32)
        usage = tf.one_hot([0]*batch_size, self._memory_size_pos, axis=-1, dtype=tf.float32)
        memory = tf.zeros([batch_size, self._memory_size_pos]+self._slot_size, tf.float32)
        access_state_pos = AccessStatePos(init_memory=init_feature,
                                   memory=memory,
                                   read_weight=read_weight,
                                   write_weight=write_weights,
                                   control_factors=control_factors,
                                   write_decay = write_decay,
                                   usage=usage)


        memory_neg = tf.zeros([batch_size, self._memory_size_neg] + self._slot_size, tf.float32)
        write_weights_neg = tf.zeros([batch_size, config.neg_num_write, self._memory_size_neg], tf.float32)
        usage_neg = tf.zeros([batch_size, self._memory_size_neg], tf.float32)
        access_state_neg = AccessStateNeg(memory=memory_neg,
                                          write_weight=write_weights_neg,
                                          usage=usage_neg)

        neg_template = tf.zeros([batch_size, config.neg_num_write] + self._slot_size, tf.float32)
        neg_idx = tf.zeros([batch_size, config.neg_num_write], tf.int32)
        return MemNetState(controller_state=controller_state, access_state_pos=access_state_pos,
                           access_state_neg=access_state_neg, neg_template=neg_template, neg_idx=neg_idx)

    @property
    def state_size(self):
        return MemNetState(controller_state=self._controller.state_size, access_state_pos=self._memory_access_pos.state_size,
                           access_state_neg=self._memory_access_neg.state_size, neg_template=self._slot_size)

    @property
    def output_size(self):
        return tf.TensorShape(self._slot_size)


if __name__=='__main__':
    from matplotlib import pyplot as plt
    x, y = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
    sigma = 0.2
    mu = [0,0]
    d = np.sqrt((x-mu[0])**2 + (y-mu[1])**2)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

    mu = [0.3, 0.5]
    d = np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2)
    g1 = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

    mu = [-0.5, 0.3]
    d = np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2)
    g2 = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))

    response = g+g1*0.5+0.8*g2
    plt.imshow(response)
    search_feature = np.random.rand(1, 22,22,256)
    neg_templates, neg_idx = extract_neg_template(np.expand_dims(response, 0), search_feature, 2)

    for idx in neg_idx[0]:
        x = idx%response.shape[1]
        y = idx//response.shape[1]
        neg_show = plt.Circle((x, y), radius=1, edgecolor="red")
        plt.gca().add_patch(neg_show)

    plt.pause(100)