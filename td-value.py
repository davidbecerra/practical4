from __future__ import division
import numpy as np
import numpy.random as npr
import sys
from SwingyMonkey import SwingyMonkey
import matplotlib.pyplot as plt

class Learner:

  def __init__(self, epsilon_factor = 10.0):
    self.last_state  = None
    self.last_action = None
    self.last_reward = None
    # TD-learning
    self.N_sa = {} # state hash -> [a0, a1]
    self.N_sas = {} # state hash -> state' hash -> [a0, a1]
    self.R = {} # state hash -> [a0, a1]
    self.V = {} # state hash -> int
    # discount used in V update
    self.discount = 0.0

    self.epsilon_factor = epsilon_factor

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
    bin_state[0] = (state['tree']['top'] - 200) // 25

    tree_dist = state['tree']['dist']
    if tree_dist < 0:
      bin_state[1] = 0
    elif tree_dist < 300:
      bin_state[1] = (tree_dist // 75) + 1
    else:
      bin_state[1] = 5

    monkey_top = state['monkey']['top']
    if monkey_top < 125:
      bin_state[2] = 0
    elif monkey_top < 350:
      bin_state[2] = ((monkey_top - 125) // 25) + 1 
    else:
      bin_state[2] = 10

    monkey_vel = state['monkey']['vel']
    if monkey_vel < 0:
      bin_state[3] = 0
    elif monkey_vel < 5:
      bin_state[3] = 1
    else:
      bin_state[3] = 2

    return bin_state

  def state_hash(self, state):
    a, b, c, d = self.state_tupler(state)
    return int((a*1000000)+(b*10000)+(c*100)+(d))

  def update_V(self, new_state):
    s = self.state_hash(self.last_state)
    s_prime = self.state_hash(new_state)
    alpha = 1.0 / self.N_sa[s][self.last_action]
    if s not in self.V:
      self.V[s] = 0.0
    if s_prime not in self.V:
      self.V[s_prime] = 0.0
    self.V[s] = self.V[s] + alpha*(self.last_reward 
                      + self.discount*self.V[s_prime] - self.V[s])

  def update_N(self, state):
    ''' Updates N(s,a) matrix and N(s,a,s') where s' = state, s = last_state and
    a = last_acion '''
    s = self.state_hash(self.last_state)
    s_prime = self.state_hash(state)
    if s not in self.N_sa:
      self.N_sa[s] = np.array([0.0, 0.0])
    if s not in self.N_sas:
      self.N_sas[s] = {}
    if s_prime not in self.N_sas[s]:
      self.N_sas[s][s_prime] = np.array([0.0, 0.0])
    self.N_sa[s][self.last_action] += 1.0
    self.N_sas[s][s_prime][self.last_action] += 1.0

  def optimal_action(self, state):
    '''
    Determines the optimal action to take in 'state' based on current model:
      pi(s) = argmax_a[ R(s,a) + sum_s'{ P(s'| s,a) * V(s') } ]
    Input: state: current state monkey is in. 
    Output: Action (0,1) based on above equation
    '''
    s = self.state_hash(state)
    # First time in state: R(s,a) = 0, N(s,a) = 0, P(s'|s,a) = 0. Pick rand action
    if s not in self.N_sa:
      return npr.rand() < 0.1
    policy = []
    for action in [0,1]:
      # Never done action in s: R(s,a) = 0, N(s,a,s') = 0, P(s'|s,a) = 0
      if self.N_sa[s][action] == 0.0:
        policy.append(0.0)
        continue
      expected_R = self.R[s][action] / self.N_sa[s][action]
      expected_V = 0.0
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

    epsilon = 1.0 / ((ii + 1.0) * self.epsilon_factor)

    # First turn of game -> pick a random action
    if self.last_action == None:
      if ii == 0.0:
        new_action = npr.rand() < 0.5
      # Use model when it exists (not on first epoch)
      else: 
        new_action = self.optimal_action(state)
    # Update V and N, then pick new action
    else: 
      self.update_N(state)
      self.update_V(state)
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

    # Update reward
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

# Display plot of scores
domain = np.arange(1, iters + 1, 1)
plt.plot(domain, scores)
plt.title("Scores over each Epoch (discount = " + str(learner.discount) + ")")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("td_scores.png")
plt.show()

    
