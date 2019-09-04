import tensorflow as tf
import numpy as np

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.build()

    def build(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], 'state')
        self.a = tf.placeholder(tf.int32, None, 'action')
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')

        n_l1, w_initializer, b_initializer = 20, tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        with tf.variable_scope('actor'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope('prob'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
                self.action_prob = tf.nn.softmax(tf.matmul(l1, w2) + b2)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.action_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.action_prob, {self.s:s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict={self.s:s, self.a:a, self.td_error:td})
        return exp_v


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, reward_decay=0.9):
        self.sess = sess
        self.gamma = reward_decay
        self.lr = lr
        self.n_features = n_features
        self.build()

    def build(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], 'state')
        self.v_next = tf.placeholder(tf.float32, [1, 1], 'v_next')
        self.r = tf.placeholder(tf.float32, None, 'r')

        n_l1, w_initializer, b_initializer = 20, tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        with tf.variable_scope('critic'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope('v'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer)
                self.v = tf.nn.softmax(tf.matmul(l1, w2) + b2)

        with tf.variable_scope('td_error'):
            self.td_error = self.r + self.gamma * self.v_next - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_next):
        s, s_next = s[np.newaxis, :], s_next[np.newaxis, :]
        v_next = self.sess.run(self.v, {self.s:s_next})
        _, td_error = self.sess.run([self.train_op, self.td_error], feed_dict={self.s:s, self.v_next:v_next, self.r:r})





