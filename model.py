import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from MuZero.nets import Nets
from MuZero.tree import Tree
from tqdm import tqdm

class MOdel(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 rp_shape,
                 batch_size=32,
                 memory_size=1000,
                 length=5,
                 discount=0.9,
                 c1=10,
                 c2=1000,
                 num_simulations=30,
                 temperature=0.9,
                 MAX_T=20):
        super().__init__()
        self.state_shape = state_shape
        self.state_shape_ = state_shape[0] * state_shape[1] * state_shape[2]
        self.action_shape = action_shape
        self.rp_shape = rp_shape
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.discount = discount
        self.nets = Nets(state_shape, action_shape, rp_shape)
        self.tree = Tree(self.nets, self.action_shape, length, discount, c1, c2)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_num = 0

        self.MAX_T = MAX_T
        self.memory = np.zeros((self.memory_size, self.MAX_T, self.action_shape * 2 + 2))
        self.observation = np.zeros((self.memory_size, self.state_shape_))
        self.max_t = np.zeros((self.memory_size, 1), dtype=int)
        self.l_history = []
        self.p_history = []
        self.v_history = []
        self.r_history = []

    def sample_batch_data(self):
        if self.batch_size > self.memory_num:
            max_t = np.max(self.max_t)
            batch_data = torch.tensor(self.memory[:self.memory_num], dtype=torch.float32)
            observation = torch.tensor(self.observation[:self.memory_num], dtype=torch.float32)
        else:
            indexes = np.random.choice(self.memory_num, size=self.batch_size)
            max_t = np.max(self.max_t[indexes])
            batch_data = torch.tensor(self.memory[indexes], dtype=torch.float32)
            observation = torch.tensor(self.observation[indexes], dtype=torch.float32)
        return max_t, batch_data[:, :max_t, :], observation

    def add_data(self, data, T):
        if self.memory_num >= self.memory_size:
            index = np.random.randint(0, self.memory_size)
            self.memory[index, :T] = data[:, self.state_shape_:]
            self.observation[index] = data[0, :self.state_shape_]
            self.max_t[index] = T
        else:
            self.memory[self.memory_num, :T] = data[:, self.state_shape_:]
            self.observation[self.memory_num] = data[0, :self.state_shape_]
            self.max_t[self.memory_num] = T
            self.memory_num += 1

    def add_traj(self, trajs):
        T = len(trajs)
        data = np.zeros((T, self.state_shape_ + 2 * self.action_shape + 2))
        g = 0
        for id, traj in enumerate(reversed(trajs)):
            observation, action, reward, pi = np.array(traj[0]).transpose(2, 1, 0).reshape(-1), \
                                              np.array(np.eye(self.action_shape)[traj[1]]).reshape(-1), \
                                              np.array(traj[2]).reshape(-1), \
                                              np.array(traj[3]).reshape(-1)
            g = self.discount * g + reward
            data[T - id - 1, :] = np.concatenate([observation, action, reward, g, pi])

        if T <= self.MAX_T:
            self.add_data(data, T)
        else:
            for i in range(T - self.MAX_T):
                self.add_data(data[i:i + self.MAX_T, :], self.MAX_T)

    def plan(self, observation):
        pi = self.tree.plan(observation, self.num_simulations, self.temperature)
        action = np.random.choice(a=self.action_shape, size=1, replace=False, p=pi)
        return action, pi

    def plan_with_mask(self, observation, mask):
        pi = self.tree.plan(observation, self.num_simulations, self.temperature) * mask
        pi = pi / np.sum(pi)
        action = np.random.choice(a=self.action_shape, size=1, replace=False, p=pi)
        return action, pi

    def learn(self, num_iter):
        optimizer = optim.SGD(self.nets.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.75)
        for _ in tqdm(range(num_iter)):
            p_loss, v_loss, r_loss = 0, 0, 0
            self.nets.train()
            T, data, observation = self.sample_batch_data()
            observation, action, reward, g, pi = observation.reshape(-1, self.state_shape[0], self.state_shape[1],
                                                                     self.state_shape[2]), \
                                                 data[:, :, :self.action_shape], \
                                                 data[:, :, self.action_shape:self.action_shape + 1], \
                                                 data[:, :, self.action_shape + 1:self.action_shape + 2], \
                                                 data[:, :, self.action_shape + 2:self.action_shape * 2 + 2]
            rp = self.nets.representation(observation)
            for t in range(T):
                p, v = self.nets.prediction(rp)
                rp, r = self.nets.dynamics(rp, action[:, t])
                p_loss = p_loss + torch.sum(- pi[:, t] * torch.log(p))
                r_loss = r_loss + torch.sum((reward[:, t] - r) ** 2)
                v_loss = v_loss + torch.sum((g[:, t] - v) ** 2)

            loss = (p_loss + r_loss + v_loss) / (self.batch_size * T)
            # print("[loss]: ", loss.data, "  [p_loss]: ", p_loss.data, "  [v_loss]: ", v_loss.data, "  [r_loss]: ", r_loss.data)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.l_history.append(loss.data)
            self.p_history.append(p_loss.data / (self.batch_size * T))
            self.v_history.append(v_loss.data / (self.batch_size * T))
            self.r_history.append(r_loss.data / (self.batch_size * T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def plot_loss(self):
        from matplotlib import pyplot as plt
        plt.figure()
        l, = plt.plot(self.l_history)
        p, = plt.plot(self.p_history)
        v, = plt.plot(self.v_history)
        r, = plt.plot(self.r_history)
        plt.legend((l, p, v, r),('total_loss', 'prob_loss', 'value_loss', 'reward_loss'))
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.show()
        
    def save_model(self, path="MuZero/c4"):
        torch.save(self.nets.dynamics, path+"/d.pkl")
        torch.save(self.nets.representation, path+"/r.pkl")
        torch.save(self.nets.prediction, path+"/p.pkl")