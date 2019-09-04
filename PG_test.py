from env.maze_env import Maze
from PGmodel.PolicyGradient import PolicyGradient as RLmodel

def maze():
    step = 0
    for episode in range(300):
        print(episode)
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            observation = observation_
            RL.store_transition(observation, action, float(reward))
            if done:
                break
            step += 1
        #RL.learn()
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = RLmodel(env.n_actions, env.n_features)
    RL.saver.restore("PGmodel/PG/PG.ckpt")
    env.after(100, maze)
    env.mainloop()