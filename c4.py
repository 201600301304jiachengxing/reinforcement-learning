import numpy as np
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

