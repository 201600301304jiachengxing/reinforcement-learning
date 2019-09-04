import tensorflow as tf
import numpy as np
from env.maze_env import Maze
import multiprocessing
import shutil
import threading

class ACnet:
    def __init__(self, n_features, n_actions, scope, beta=0.001, globalAC=None):
        self.n_features = n_features
        self.n_actions = n_actions

        if scope == 'GLOBAL_NET':
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_features], 's')
                self.a_params, self.c_params = self.build(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_features], 's')
                self.a_history = tf.placeholder(tf.float32, [None,], 'a')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'v')

                self.a_prob, self.v, self.a_params, self.c_params = self.build(scope)
                td = tf.subtract(self.v_target, self.v, name='TD')
                with tf.name_scope('c_loss'):
                    self.loss_c = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_history, self.n_actions, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keep_dims=True)
                    self.exp_v = beta * entropy + exp_v
                    self.loss_a = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.grads_a = tf.gradients(self.loss_a, self.a_params)
                    self.grads_c = tf.gradients(self.loss_c, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.grads_a, globalAC.a_params))
                    self.update_c_op = opt_c.apply_gradients(zip(self.grads_c, globalAC.c_params))

    def build(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, self.n_actions, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)

    def pull_global(self):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        prob_weights = sess.run(self.a_prob, feed_dict={self.s:s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = Maze()
        self.AC = ACnet(name, globalAC)
        self.name = name

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        for episode in range(300):
            print(episode)
            observation = self.env.reset()
            ep_r = 0
            while True:
                self.env.render()
                action = self.AC.choose_action(observation)
                observation_, reward, done = self.env.step(action)
                ep_r += reward

                buffer_s.append(observation)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % 10 ==0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.v, feed_dict={self.AC.s:observation_[np.newaxis, :]})[0,0]
                    buffer_v_target = []

                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_history: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                observation = observation_
                total_step += 1
                if done:
                    break


GAMMA = 0.9
sess = tf.Session()

with tf.device('/cpu:0'):
    opt_a = tf.train.RMSPropOptimizer(0.001)
    opt_c = tf.train.RMSPropOptimizer(0.001)
    global_ac = ACnet('GLOBAL_NET')

    workers = []

    for i in range(4):
        i_name = 'w_%i' % i
        workers.append(Worker(i_name, global_ac))

coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())

worker_threads = []
for worker in workers:
    job = lambda: worker.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)

