from env.maze_env import Maze
from DQNmodel.DQN import DQN as RLmodel

def maze(art=False):
    step = 0
    for episode in range(300):
        print(episode)
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            if art:
                if all(observation == observation_):
                    reward = -0.01
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
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
    env.after(100, maze)
    env.mainloop()
    RL.plot_cost()
    RL.save_model("DQNmodel/DQN/DQN.ckpt")