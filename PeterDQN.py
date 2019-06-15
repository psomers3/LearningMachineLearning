from lib.PeterNN import DQN
import numpy as np
import gym


env = gym.make('CartPole-v0')
sim_step_limit = 1500
env._max_episode_steps = sim_step_limit
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

layer_specs = list()
layer_specs.append({'layer': 'dense',
                           'units': 8,
                           'activation': 'relu'})
layer_specs.append({'layer': 'dense',
                           'units': 8,
                           'activation': 'relu'})
layer_specs.append({'layer': 'dense',
                           'units': action_dim,
                           'activation': 'softmax'})

myAgent = DQN(input_shape=(None, observation_dim), layers=layer_specs)
myAgent.initiate()
myAgent.set_to_learn(True)

epoc_length = 100

k = 0
total_num_steps = 0
try:
    while True:
        for i in range(epoc_length):
            s = env.reset()
            for j in range(sim_step_limit):
                # Probabilistically pick an action given our network outputs.
                a = myAgent.make_decision(s)

                s1, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
                # s1 = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
                reward = 1/np.linalg.norm([s1[0]*50, s1[1]*10, s1[2]*100, s[3]*10])
                myAgent.log_result(s, a, reward)
                s = s1
                if done:
                    break

            myAgent.update()

            total_num_steps += j
        print('epoch ', k, ', Avg num steps alive: ', total_num_steps/epoc_length)
        if total_num_steps/epoc_length > (0.95*sim_step_limit):
            break
        total_num_steps = 0
        k += 1
except:
    pass

s = env.reset()
myAgent.set_to_learn(False)
input('Enter any key to start playing result:')
while True:
    # Probabilistically pick an action given our network outputs.
    a = myAgent.make_decision(s)
    s, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
    env.render()
    if done:
        break
env.close()