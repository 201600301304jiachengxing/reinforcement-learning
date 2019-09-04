import tensorflow as tf
import numpy as np

class DDPG:
    def __init__(self, n_features, n_actions, capacity=4000, TAU=0.01, lr_a=1e-3, lr_c=2e-3, gamma=0.9, batch_size=32):
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_actions = n_actions
        self.capacity = capacity
        self.memory = np.zeros((self.capacity, self.n_features * 2 + self.n_actions + 1), dtype=np.float32)
        self.pointer = 0

        self.sess = tf.Session()

        self.s = tf.placeholder(tf.float32, [None, self.n_features], 's')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], 's')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self.build_a(self.s)
        self.q = self.build_c(self.s, self.a)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        self.a_ = self.build_a(self.s_, reuse=True, custom_getter=ema_getter)
        self.q_ = self.build_c(self.s_, self.a_, reuse=True, custom_getter=ema_getter)

        self.loss_a = - tf.reduce_mean(self.q)
        self.train_op_a = tf.train.AdamOptimizer(lr_a).minimize(self.loss_a)

        with tf.control_dependencies(target_update):
            q_target = self.r + gamma * self.q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.train_op_c = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.n_actions, activation=tf.nn.softmax, name='a', trainable=trainable)
            return a

    def build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.s: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.n_features]
        ba = bt[:, self.n_features:self.n_features+self.n_actions]
        br = bt[:, -self.n_features-1:-self.n_features]
        bs_ = bt[:, -self.n_features:]

        self.sess.run(self.train_op_a, feed_dict={self.s:bs})
        self.sess.run(self.train_op_c, feed_dict={self.s:bs,self.a:ba,self.r:br,self.s_:bs_})

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path=save_path)

