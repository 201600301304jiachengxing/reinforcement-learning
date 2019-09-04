import tensorflow as tf
import numpy as np

class PPO:
    def __init__(self, n_features, n_actions, batch_size=32, lr_a=0.001, lr_c=0.002, step_a=10, step_c=10, gamma=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.step_a = step_a
        self.step_c = step_c

        self.method = [dict(name='kl_pen', kl_target=0.01, lam=0.5), dict(name='clip', epsilon=0.2)][1]

        self.sess = tf.Session()

        self.s = tf.placeholder(tf.float32, [None, self.n_features], 'state')
        self.r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.advantage = self.r - self.v
            self.loss_c = tf.reduce_mean(tf.square(self.advantage))
            self.train_op_c = tf.train.AdamOptimizer(lr_c).minimize(self.loss_c)

        pi, pi_params = self.build_a('pi', trainable=True)
        pi_old, pi_params_old = self.build_a('pi_old', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_pi_old'):
            self.update_pi_old_op = [p_old.assign(p) for p, p_old in zip(pi_params, pi_params_old)]

        self.a = tf.placeholder(tf.float32, [None, self.n_actions], 'action')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.a) / pi_old.prob(self.a)
                surr = ratio * self.adv
            if self.method['name']=='kl_pen':
                self.lamb = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(pi_old, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.loss_a = - tf.reduce_mean(surr - self.lamb * kl)
            else:
                self.loss_a = - tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.-self.method['epsilon'], 1.+self.method['epsilon']) * self.adv))

        with tf.variable_scope('train_a'):
            self.train_op_a = tf.train.AdamOptimizer(lr_a).minimize(self.loss_a)

        self.sess.run(tf.global_variables_initializer())

    def build_a(self, pi_name, trainable):
        with tf.variable_scope(pi_name):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.n_actions, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.n_actions, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=pi_name)
        return norm_dist, params

    def update(self, s, a, r):
        self.sess.run(self.update_pi_old_op)
        adv = self.sess.run(self.advantage, {self.s:s, self.r:r})

        if self.method['name'] == 'kl_pen':
            kl = None
            for _ in range(self.step_a):
                _, kl = self.sess.run([self.train_op_a, self.kl_mean], {self.s:s,self.a:a,self.adv:adv,self.lamb:self.method['lam']})
                if kl > self.method['kl_target']:
                    break
            if kl < self.method['kl_target'] / 1.5:
                self.method['lam'] /= 2
            elif kl > self.method['kl_target'] * 1.5:
                self.method['lam'] *= 2

            self.method['lam'] = np.clip(self.method['lam'], 1e-4, 10)
        else:
            [self.sess.run(self.train_op_a, {self.s:s,self.a:a,self.adv:adv}) for _ in range(self.step_a)]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.s:s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.s:s})[0,0]

