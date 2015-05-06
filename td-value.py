from __future__ import division
import numpy as np
import numpy.random as npr
import sys
from SwingyMonkey import SwingyMonkey
import matplotlib.pyplot as plt

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        # state indexing in order: tree top, tree dist, monkey top, monkey vel, action
        # TD-learning
        self.N_sa = {} # state hash -> [a0, a1]
        self.N_sas = {} # state hash -> state' hash -> [a0, a1]
        self.R = {} # state hash -> [a0, a1]
        self.V = {} # state hash -> int
        self.a = {} # state hash -> [a0, a1]

        # discount used in Q-learning
        self.discount = 0.5

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def state_tupler(self, state):
        '''
        Input: takes full state they provide
        Return: 'tuple'/condensed representation of the state
            bin_state = [tree top bin, tree dist bin, monkey top bin, monkey vel bin]
        '''
        # state indexing in order: tree top, tree dist, monkey top, monkey vel
        bin_state = [0,0,0,0]
        bin_state[0] = (state['tree']['top'] - 200) // 50
        tree_dist = state['tree']['dist']

        if tree_dist < 0:
            bin_state[1] = 0
        elif tree_dist < 50:
            bin_state[1] = 1
        elif tree_dist < 100:
            bin_state[1] = 2
        elif tree_dist < 200:
            bin_state[1] = 3
        elif tree_dist < 300:
            bin_state[1] = 4
        else:
            bin_state[1] = 5

        monkey_top = state['monkey']['top']
        if monkey_top <= 56:
            bin_state[2] = 0
        elif not monkey_top >= 400:
            bin_state[2] = monkey_top // 50
        else:
             bin_state[2] = 8

        monkey_vel = state['monkey']['vel']
        if monkey_vel < -25:
            bin_state[3] = 0
        elif not monkey_vel >= 25:
            bin_state[3] = (monkey_vel + 30) // 5
        else:
            bin_state[3] = 11

        return bin_state

    def state_hash(self, state):
        a, b, c, d = self.state_tupler(state)
        return int(str(a*1000000)+str(b*10000)+str(c*100)+str(d))

    def update_V(self, new_state):
        s = self.state_hash(self.last_state)
        s_prime = self.state_hash(new_state)
        if s not in self.a:
            self.a[s] = np.array([1.0,1.0])
        alpha = 1.0 / self.a[s][self.last_action]
        if s not in self.V:
            self.V[s] = 0.0
        if s_prime not in self.V:
            self.V[s_prime] = 0.0
        self.V[s] = self.V[s] + alpha*(self.last_reward + self.discount*self.V[s_prime] - self.V[s])

    def update_a(self):
        s = self.state_hash(self.last_state)
        self.a[s][self.last_action] += 1.0

    def update_transition_model(self, state):
        s = self.state_hash(self.last_state)
        if s not in self.N_sa:
            self.N_sa[s] = np.array([0.0, 0.0])
        self.N_sa[s][self.last_action] += 1.0

        s_prime = self.state_hash(state)
        if s not in self.N_sas:
            self.N_sas[s] = {}
        if s_prime not in self.N_sas[s]:
            self.N_sas[s][s_prime] = np.array([0.0, 0.0])
        self.N_sas[s][s_prime][self.last_action] += 1.0

    def optimal_action(self, state):
        s = self.state_hash(state)
        policy = []
        for action in [0,1]:
            if s not in self.R:
                expected_R = 0
            else:
                expected_R = self.R[s][action] / self.N_sa[s][action]
            expected_V = 0.0
            if s in self.N_sas:
                for s_prime in self.N_sas[s]:
                    P = self.N_sas[s][s_prime][action] / self.N_sa[s][action]
                    expected_V += P * self.V[s_prime]
            policy.append(expected_R + expected_V)
        return np.argmax(policy)

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.
        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        # tree dist ranges from 400 to negatives
        # tree top ranges from 400 to 200
        # tree bot ranges from 200 to 0
        # gap always 200 pixels
        # monkey 56 pixels tall
        # monkey between 450 and -50
        # monkey vel between -50 and 40

        epsilon = 1.0 / (10.0 * (ii+1.0))

        # First turn of game -> just pick a random action
        if self.last_action == None:
            new_action = npr.rand() < 0.5
        # Update Q and a, then pick new action
        else: 
            self.update_transition_model(state)
            self.update_V(state)
            self.update_a()
            new_action = self.optimal_action(state)
            # Explore (take non-optimal action) with probability epsilon
            if npr.rand() < epsilon:
                new_action = int(not new_action)

        self.last_action = new_action
        self.last_state  = state

        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
        if self.last_state != None:
            s = self.state_hash(self.last_state)
            if s not in self.R:
                self.R[s] = np.array([0.0, 0.0])
            self.R[s][self.last_action] += reward

iters = 300
learner = Learner()
scores = []

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    scores.append(swing.get_score())

    # Reset the state of the learner.
    learner.reset()

domain = np.arange(1, iters + 1, 1)
plt.plot(domain, scores)
# plt.bar(domain, scores)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("scores.png")
plt.show()

    
