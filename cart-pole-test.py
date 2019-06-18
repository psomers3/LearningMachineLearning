import lib.PeterNN as NN
import gym

env = gym.make('CartPole-v0')
sim_step_limit = 1500
env._max_episode_steps = sim_step_limit

new_agent = NN.create_from_json('./DQN_test.json')
new_agent.set_to_learn(False)

s = env.reset()
while True:
    # Probabilistically pick an action given our network outputs.
    a = new_agent.make_decision(s)
    s, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
    env.render()
    if done:
        break
env.close()