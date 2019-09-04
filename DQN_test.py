from env.maze_env import Maze
from DQNmodel.DQN import DQN as RLmodel

def maze():
    step = 0
    for episode in range(300):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action_test(observation)
            observation_, reward, done = env.step(action)
            observation = observation_
            if done:
                break
            step += 1
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = RLmodel(env.n_actions, env.n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=2000
                 )
    RL.saver.restore(RL.sess, "DQNmodel/DQN/DQN.ckpt")
    env.after(100, maze)
    env.mainloop()