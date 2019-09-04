import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 sess=None):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.e_greedy
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.build()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_history = []
        self.saver = tf.train.Saver()

    def build(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.q_target = tf.placeholder = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
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
        self.q_eval = build_net('eval_net', 'eval_net_params', self.s, self.n_actions, self.n_features, n_l1)
        self.q_next = build_net('target_net', 'target_net_params', self.s_, self.n_actions, self.n_features, n_l1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s:observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action_test(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_net_replaced\n')
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index]
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_:batch_memory[:, -self.n_features:],
                                                  self.s:batch_memory[:, :self.n_features]
                                                  })
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        _, cost = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.s:batch_memory[:, :self.n_features],
                                           self.q_target:q_target})

        self.cost_history.append(cost)
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('cost')
        plt.xlabel('train step')
        plt.show()

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)



