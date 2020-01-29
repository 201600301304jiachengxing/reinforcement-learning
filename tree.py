import numpy as np

class Node:
    def __init__(self, rp, p):
        action_size = p.shape[1]
        self.rp = rp
        self.p = p
        self.n = np.zeros((action_size))
        self.r = np.zeros((action_size))
        self.q = np.zeros((action_size))
        self.subNode = [None for i in range(action_size)]

    def __repr__(self):
        return "p: {}\nn: {}\nr: {}\nq: {}".format(self.p, self.n, self.r, self.q)

class Tree:
    def __init__(self, nets, action_size, length=5, discount=0.9, c1=1.25, c2=20000):
        self.action_size = action_size
        self.nets = nets
        self.node = None
        self.length = length
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.max_q = 1
        self.min_q = 0

    def uct_action(self, node):
        n_total = np.sum(node.n)
        exploration = node.p * np.sqrt(n_total) / (1 + node.n) * (self.c1 + np.log((n_total + self.c2 + 1) / self.c2))
        exploitation = (node.q - self.min_q) / (self.max_q - self.min_q)
        uct_score = exploitation + exploration
        best_action = np.argmax(uct_score)
        return best_action

    def simulation(self, depth, node=None):
        if depth >= self.length:
            _, v = self.nets.prediction.inference(node.rp)
            return v

        action = self.uct_action(node)
        if node.subNode[action] is None:
            action_ = np.eye(self.action_size)[action]
            if len(action_.shape)==1:
                action_ = action_[np.newaxis,:]
            rp, r = self.nets.dynamics.inference(node.rp, action_)
            p, v = self.nets.prediction.inference(rp)
            node.subNode[action] = Node(rp, p)
            node.r[action] = r
            g = r + self.discount * self.simulation(depth+1, node.subNode[action])
        else:
            g = node.r[action] + self.discount * self.simulation(depth+1, node.subNode[action])
        node.q[action] = (node.n[action] * node.q[action] + g) / (node.n[action] + 1)
        node.n[action] += 1

        if node.q[action] > self.max_q:
            self.max_q = node.q[action]
        elif node.q[action] < self.min_q:
            self.min_q = node.q[action]

        return g

    def plan(self, state, num_simulations, temperature = 0.5):
        rp = self.nets.representation.inference(state)
        p, _ = self.nets.prediction.inference(rp)
        self.node = None
        self.node = Node(rp, p)
        self.max_q = 1
        self.min_q = 0
        for _ in range(num_simulations):
            self.simulation(0, self.node)

        n = np.array(self.node.n) + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()
