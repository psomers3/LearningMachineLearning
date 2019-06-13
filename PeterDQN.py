from lib.PeterNN import DQN
import numpy as np
import gym


env = gym.make('CartPole-v0')
sim_step_limit = 1500
env._max_episode_steps = sim_step_limit
observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

hidden_layer_specs = list()
hidden_layer_specs.append({'layer': 'dense',
                           'units': 8,
                           'activation': 'relu'})
hidden_layer_specs.append({'layer': 'dense',
                           'units': 8,
                           'activation': 'relu'})
hidden_layer_specs.append({'layer': 'dense',
                           'units': action_dim,
                           'activation': 'softmax'})

myAgent = DQN(input_shape=(None, observation_dim), layers=hidden_layer_specs)
myAgent.initiate()

max_ep = 1000  # Set total number of episodes to train agent on.

i = 0
total_num_steps = 0
try:
    while i < max_ep:
        s = env.reset()
        running_reward = 0

        for j in range(sim_step_limit):
            # Probabilistically pick an action given our network outputs.
            a_disc = myAgent.make_decision(s)
            a = np.random.choice(a_disc[0], p=a_disc[0])
            a = np.argmax(a_disc == a)

            s1, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
            # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
            r = 1/np.linalg.norm([s1[0]*50, s1[1]*10, s1[2]*100, s[3]*10])
            if done:
                r = -1
            myAgent.log_result(s, a, r)
            s = s1
            running_reward += r
            if done:
                myAgent.update()
                break
        if j == sim_step_limit:
            myAgent.update()

        total_num_steps += j
        if i % (max_ep * 0.05) == 0 and i != 0:
            print((i / max_ep) * 100, '% Avg num steps alive: ', total_num_steps/(max_ep*0.05))
            if total_num_steps/(max_ep*0.05) > (.95*sim_step_limit):
                break
            total_num_steps = 0

        i += 1
except:
    pass

s = env.reset()
input('Enter any key to start playing result:')
while True:
    # Probabilistically pick an action given our network outputs.
    a = np.argmax(myAgent.make_decision(s))
    s, r, done, _ = env.step(a)  # Get our reward for taking an action given a bandit.
    env.render()
    if done:
        break
env.close()