import tensorflow as tf
import numpy as np
from collections import deque
from netutil import *
import random

np.random.seed(1)
tf.set_random_seed(1)
INPUT_LENGTH_SIZE = 80
INPUT_WIDTH_SIZE = 80
INPUT_CHANNEL = 4
ACTIONS_DIM = 13

NN_UNITS = 512
MEMORY_SIZE = 50000


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=50000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize memory
        self.memory = deque(maxlen=memory_size)

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.initialize_all_variables())

        self.cost_his = []
        # self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, 'record/save_net.ckpt')

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w_c1 = tf.get_variable('w_c1', [8, 8, INPUT_CHANNEL, 32], initializer=w_initializer, collections=c_names)
                b_c1 = tf.get_variable('b_c1', [32], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(conv2d(s, w_c1, 4) + b_c1)
                h_pool1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope('l2'):
                w_c2 = tf.get_variable('w_c2', [4, 4, 32, 64], initializer=w_initializer, collections=c_names)
                b_c2 = tf.get_variable('b_c2', [64], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(conv2d(h_pool1, w_c2, 2) + b_c2)
                # h_pool2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                h_pool2 = l2

            with tf.variable_scope('l3'):
                w_c3 = tf.get_variable('w_c3', [3, 3, 64, 64], initializer=w_initializer, collections=c_names)
                b_c3 = tf.get_variable('b_c3', [64], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(conv2d(h_pool2, w_c3, 1) + b_c3)

                h_pool3 = l3
                # h_pool3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                h_conv3_out_size = np.prod(h_pool3.get_shape().as_list()[1:])
                h_conv3_flat = tf.reshape(h_pool3, [-1, h_conv3_out_size])

            with tf.variable_scope('l4'):
                w_f1 = tf.get_variable('w_f1', [h_conv3_out_size, NN_UNITS], initializer=w_initializer,
                                       collections=c_names)
                b_f1 = tf.get_variable('b_f1', [NN_UNITS], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(h_conv3_flat, w_f1) + b_f1)

            with tf.variable_scope('l5'):
                w_a_f2 = tf.get_variable('w_a_f2', [NN_UNITS, ACTIONS_DIM], initializer=w_initializer,
                                       collections=c_names)
                b_a_f2 = tf.get_variable('b_a_f2', [ACTIONS_DIM], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l4, w_a_f2) + b_a_f2

            with tf.variable_scope('Q'):
                out = self.A
            return out, h_conv3_flat
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder('float', shape=[None, INPUT_LENGTH_SIZE, INPUT_WIDTH_SIZE, INPUT_CHANNEL])  # input State
        self.s_ = tf.placeholder('float', shape=[None, INPUT_LENGTH_SIZE, INPUT_WIDTH_SIZE, INPUT_CHANNEL])  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.q_next = tf.placeholder('float', shape=[None, ACTIONS_DIM])

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.truncated_normal_initializer(mean=0.0, stddev=0.01), tf.constant_initializer(0.01)  # config of layers
            self.q_eval, self.h_conv3_flat = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #     # c_names(collections_names) are the collections to store variables
        #     c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        #     self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_wrt_a = tf.reduce_sum(self.q_eval * a_one_hot, axis=1)  # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        self.memory.append((s, a, r, s_))
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        # if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        max_Q = np.max(actions_value)
        # else:
        #     action = np.random.randint(0, self.n_actions)
        return action, max_Q

    # def _replace_target_params(self):
    #     t_params = tf.get_collection('target_net_params')
    #     e_params = tf.get_collection('eval_net_params')
    #     self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self, temp_lr):
        # check to replace target parameters
        self.lr = temp_lr
        # if self.learn_step_counter % self.replace_target_iter == 0:
        #     self._replace_target_params()
        #     print('\ntarget_params_replaced\n')

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = [t[0] for t in minibatch]
        action_batch = [t[1] for t in minibatch]
        reward_batch = [t[2] for t in minibatch]
        next_state_batch = [t[3] for t in minibatch]

        # # sample batch memory from all memory
        # if self.memory_counter > self.memory_size:
        #     sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # else:
        #     sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # batch_memory = self.memory[sample_index, :]
        temp_q_next = self.sess.run(self.q_eval, feed_dict={
            self.s: next_state_batch,
        })
        _, cost, cost1, tt, my_loss = self.sess.run(
            [self._train_op, self.q_eval_wrt_a, self.h_conv3_flat, self.q_target, self.loss],
            feed_dict={
                self.s: state_batch,
                self.a: action_batch,
                self.r: reward_batch,
                self.q_next: temp_q_next,
            })
        if self.learn_step_counter % 5000 == 0:
            self.saver.save(self.sess, 'record/save_net.ckpt')
        self.cost_his.append(my_loss)
        # print("cost")
        # print(cost)
        # print(cost1)
        # # print(cost1)
        # print('loss:', my_loss)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return my_loss

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(2, output_graph=True)



