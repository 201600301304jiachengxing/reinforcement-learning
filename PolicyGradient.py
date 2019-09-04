import tensorflow as tf
import numpy as np

class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_dacay=0.95,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_dacay

        self.observations = []
        self.actions = []
        self.rewards = []

        self.build()

        self.sess = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build(self):
        with tf.name_scope('inputs'):
            self.tf_observations = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_actions = tf.placeholder(tf.int32, [None,], name="actions")
            self.tf_rewards = tf.placeholder(tf.float32, [None,], name="rewards")

        n_l1, w_initializer, b_initializer = 10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        def build_net(net_name, param_name, placeholder, n_actions, n_features, n_l1):
            with tf.variable_scope(net_name):
                c_names = [param_name, tf.GraphKeys.GLOBAL_VARIABLES]
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1', [n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(placeholder, w1) + b1)
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', [n_l1, n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, n_actions], initializer=b_initializer, collections=c_names)
                    target = tf.matmul(l1, w2) + b2
            return target
        action = build_net('net', 'params', self.tf_observations, self.n_actions, self.n_features, n_l1)
        self.action_prob = tf.nn.softmax(action, name='action_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action, labels=self.tf_actions)
            loss = tf.reduce_mean(neg_log_prob * self.tf_rewards)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.action_prob, feed_dict={self.tf_observations:observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.observations.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def discount_norm_rewards(self):
        rewards_norm = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            rewards_norm[t] = running_add
        rewards_norm -= np.mean(rewards_norm)
        rewards_norm /= np.std(rewards_norm)
        return rewards_norm

    def learn(self):
        rewards_norm = self.discount_norm_rewards()
        self.sess.run(self.train_op, feed_dict={self.tf_observations:np.vstack(self.observations),
                                                self.tf_actions:np.array(self.actions),
                                                self.tf_rewards:rewards_norm})
        self.observations, self.actions, self.rewards = [], [], []
        return rewards_norm

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)


























