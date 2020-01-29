import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class Conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bn=False):
        super().__init__()
        self.conv = Conv(input_channel, output_channel, kernel_size, bn)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


num_filters = 8
num_blocks = 2
kernel_size = 3

class Representation(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer0 = Conv(input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters, num_filters, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.linear = nn.Linear(input_shape[1]*input_shape[2]*num_filters, output_shape)

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        shape = h.shape
        h = h.view(-1, shape[1]*shape[2]*shape[3])
        rp = self.linear(h)
        rp = F.relu(rp)
        return rp

    def inference(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            rp = self(x)
        return rp.cpu().numpy()

class Prediction(nn.Module):
    def __init__(self, rp_shape, action_shape):
        super().__init__()
        self.latent_shape = 256
        self.fc_l = nn.Linear(rp_shape, self.latent_shape)
        self.fc_p = nn.Linear(self.latent_shape, action_shape)
        self.fc_v = nn.Linear(self.latent_shape, 1)

    def forward(self, rp):
        rp_l = self.fc_l(rp)
        rp_l = F.relu(rp_l)
        h_p = self.fc_p(rp_l)
        h_v = self.fc_v(rp_l)
        return F.softmax(h_p), torch.tanh(h_v)

    def inference(self, rp):
        self.eval()
        rp = torch.tensor(rp, dtype=torch.float32)
        with torch.no_grad():
            p, v = self(rp)
        return p.cpu().numpy(), v.cpu().numpy()

class Dynamics(nn.Module):
    def __init__(self, rp_shape, action_shape):
        super().__init__()
        self.input_shape = rp_shape + action_shape
        self.latent_shape = 1024
        self.fc_l = nn.Linear(self.input_shape, self.latent_shape)
        self.fc_r = nn.Linear(self.latent_shape, 1)
        self.fc_h = nn.Linear(self.latent_shape, rp_shape)

    def forward(self, rp, a):
        input = torch.cat([rp, a], dim=1)
        l = self.fc_l(input)
        l = F.relu(l)
        h = self.fc_h(l)
        r = self.fc_r(l)
        return F.relu(h), torch.sigmoid(r)

    def inference(self, rp, a):
        self.eval()
        rp = torch.tensor(rp, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        with torch.no_grad():
            rp, r = self(rp, a)
        return rp.cpu().numpy(), r.cpu().numpy()

class Nets(nn.Module):
    def __init__(self, state_shape, action_shape, rp_shape):
        super().__init__()
        self.representation = Representation(state_shape, rp_shape)
        self.prediction = Prediction(rp_shape, action_shape)
        self.dynamics = Dynamics(rp_shape, action_shape)

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

    def uct_action(self, node, dirichlet=False):
        n_total = np.sum(node.n)
        if dirichlet:
            p = 0.9 * node.p + 0.1 * np.random.dirichlet([0.15] * len(node.p))
            uct_score = node.q + p \
                        * np.sqrt(n_total) / (1 + node.n) \
                        * self.c1 + np.log((n_total + self.c2 + 1) / self.c2)
        else:
            uct_score = node.q + node.p \
                        * np.sqrt(n_total) / (1 + node.n) \
                        * self.c1 + np.log((n_total + self.c2 + 1) / self.c2)
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

        return g

    def plan(self, state, num_simulations, temperature = 0.5):
        rp = self.nets.representation.inference(state)
        p, _ = self.nets.prediction.inference(rp)
        self.node = None
        self.node = Node(rp, p)
        for _ in range(num_simulations):
            self.simulation(0, self.node)

        n = np.array(self.node.n) + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()


class MOdel(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 rp_shape,
                 batch_size=128,
                 memory_size=2000,
                 length=5,
                 discount=0.9,
                 c1=1.25,
                 c2=20000,
                 num_simulations=20,
                 temperature=0.5,
                 MAX_T=10):
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

    def save_model(self, path=""):
        torch.save(self.nets.dynamics, path + "d.pkl")
        torch.save(self.nets.representation, path + "r.pkl")
        torch.save(self.nets.prediction, path + "p.pkl")


import sys, time
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class C4(tk.Tk, object):
    def __init__(self, visual=False):
        super(C4, self).__init__()
        self.row, self.column = 6, 7

        self.n_actions = self.column
        self.n_features = self.row * self.column

        self.board = np.zeros((self.row, self.column))
        self.high = [0 for _ in range(self.column)]

        self.player = 1
        self.point = []

        self.mesh = 25
        self.ratio = 0.9
        self.board_color = "#CDBA96"
        self.header_bg = "#CDC0B0"
        self.btn_font = ("黑体", 12, "bold")
        self.step_ = self.mesh / 2
        self.chess_r = self.step_ * self.ratio
        self.point_r = self.step_ * 0.2
        self.color = {-1:'red',1:'blue'}

        self.visual = visual
        if visual:
            self.build_board()

    def reset(self):
        self.board = np.zeros((self.row, self.column))
        self.high = [0 for _ in range(self.column)]
        self.player = 1
        for p in self.point:
            self.c4.delete(p)
        if self.visual:
            self.draw_board()
        return self.board

    def step(self, action):
        action = action[0]

        self.board[self.row - self.high[action] - 1][action] = self.player
        self.high[action] += 1

        if self.is_win(self.row - self.high[action], action, self.player):
            reward = self.player
            done = True
        else:
            reward = 0
            if np.sum(self.board==0)==0:
                done = True
            else:
                done = False

        if self.visual:
            self.draw_chess(self.row - self.high[action], action, self.color[self.player])
        self.player = - self.player
        return self.board, reward, done

    def render(self):
        # time.sleep(2)
        self.update()

    def valid_mask(self):
        mask = np.ones((self.column))
        for i in range(self.column):
            if self.high[i]>=self.row:
                mask[i] = 0
        return mask

    def is_win(self, x, y, tag):
        def direction(i, j, di, dj, row, column, matrix):
            temp = []
            while 0 <= i < row and 0 <= j < column:
                i, j = i + di, j + dj
            i, j = i - di, j - dj
            while 0 <= i < row and 0 <= j < column:
                temp.append(matrix[i][j])
                i, j = i - di, j - dj
            return temp

        four_direction = []
        four_direction.append([self.board[i][y] for i in range(self.row)])
        four_direction.append([self.board[x][j] for j in range(self.column)])
        four_direction.append(direction(x, y, 1, 1, self.row, self.column, self.board))
        four_direction.append(direction(x, y, 1, -1, self.row, self.column, self.board))

        for v_list in four_direction:
            count = 0
            for v in v_list:
                if v == tag:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def build_board(self):
        ###########
        #   GUI   #
        #######################################
        self.title("C4game")
        self.resizable(width=False, height=False)
        self.header = tk.Frame(self, highlightthickness=0, bg=self.header_bg)
        self.header.pack(fill=tk.BOTH, ipadx=10)
        self.c4 = tk.Canvas(self, bg=self.board_color, width=(self.column + 1) * self.mesh,
                              height=(self.row + 1) * self.mesh, highlightthickness=0)
        self.draw_board()
        self.c4.pack()
        #self.mainloop()

    def draw_mesh(self, x, y):
        ratio = (1 - self.ratio) * 0.99 + 1
        center_x, center_y = self.mesh * (x + 1), self.mesh * (y + 1)
        self.c4.create_rectangle(center_y - self.step_, center_x - self.step_,
                                      center_y + self.step_, center_x + self.step_,
                                      fill=self.board_color, outline=self.board_color)
        a, b = [0, ratio] if y == 0 else [-ratio, 0] if y == self.column - 1 else [-ratio, ratio]
        c, d = [0, ratio] if x == 0 else [-ratio, 0] if x == self.row - 1 else [-ratio, ratio]
        self.c4.create_line(center_y + a * self.step_, center_x, center_y + b * self.step_, center_x)
        self.c4.create_line(center_y, center_x + c * self.step_, center_y, center_x + d * self.step_)

    def draw_chess(self, x, y, color):
        center_x, center_y = self.mesh * (x + 1), self.mesh * (y + 1)
        p = self.c4.create_oval(center_y - self.chess_r, center_x - self.chess_r,
                                center_y + self.chess_r, center_x + self.chess_r,
                                fill=color)
        self.point.append(p)

    def draw_board(self):
        [self.draw_mesh(x, y) for y in range(self.column) for x in range(self.row)]

    def __str__(self):
        print(self.board)


def C4_game():
    step = 0
    global s_shape
    for episode in range(3000):
        traj = []
        t = 0
        #print('no.: ',episode)
        observation = env.reset()
        ob = np.zeros(s_shape)
        while True:
            mask = env.valid_mask()
            ob[0:3] = ob[1:4]
            ob[3] = np.array([observation])
            action, pi = RL.plan_with_mask(ob[np.newaxis,:,:,:], mask)
            observation_, reward, done = env.step(action)
            env.render()

            if done:
                print('[episode]: ', episode, '|| [result]: ',reward)
                traj.append((ob, action, reward, pi))
                RL.add_traj(traj)
                break

            mask_ = env.valid_mask()
            ob[0:3] = ob[1:4]
            ob[3] = np.array([observation_])
            action_ , pi_ = RL.plan_with_mask(ob[np.newaxis,:,:,:], mask_)
            observation_, reward, done = env.step(action_)
            env.render()

            traj.append((ob, action, reward, pi))
            observation = observation_
            if done:
                RL.add_traj(traj)
                print('[episode]: ', episode, '|| [result]: ', reward)
                break
            t += 1
            step += 1

            RL.learn(1)
            if episode % 500 == 0:
                RL.save_model()
                RL.plot_loss()
            print('learn')



    print('game over')


if __name__ == "__main__":
    # C4 game
    env = C4(False)
    s_shape = (4,6,7)
    a_shape = 7
    r_shape = 250
    RL = MOdel(s_shape, a_shape, r_shape)
    C4_game()
    env.mainloop()
    env.close()