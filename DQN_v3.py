import tensorflow as tf
import numpy as np

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_idx = 2 * parent_idx - 1
            right_idx = left_idx + 1
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    v -= self.tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]


class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, weights, = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        priority_seg = self.tree.total_p() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
        for i in range(n):
            a, b = priority_seg * i, priority_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            weights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, weights


class PrioritizedDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0

        self.prioritized = prioritized
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.build()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_history = []
        self.saver = tf.train.Saver()

    def build(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.q_target = tf.placeholder = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        if self.prioritized:
            self.weight = tf.placeholder(tf.float32, [None, 1], name='weight')

        n_l1, w_initializer, b_initializer = 20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

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
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)
                self.loss = tf.reduce_mean(self.weight * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
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

        if self.prioritized:
            tree_idx, batch_memory, weights = self.memory.sample(self.batch_size)
        else:
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

        if self.prioritized:
            _, abs_errors, cost = self.sess.run([self.train_op, self.abs_errors, self.loss],
                                                feed_dict={self.s: batch_memory[:, :self.n_features],
                                                           self.q_target: q_target,
                                                           self.weight: weights})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, cost = self.sess.run([self.train_op, self.loss],
                                    feed_dict={self.s: batch_memory[:, :self.n_features],
                                               self.q_target: q_target})
        self.cost_history.append(cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.e_greedy
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('cost')
        plt.xlabel('train step')
        plt.show()

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)



























