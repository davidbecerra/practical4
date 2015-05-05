from __future__ import division
import numpy.random as npr
import sys
from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        # state indexing in order: tree top, tree dist, monkey top, monkey vel, action
        self.Q = [[[[[0]*2]*12]*9]*6]*4

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def state_tupler(state):
        # state indexing in order: tree top, tree dist, monkey top, monkey vel, action
        bin_state = (0,0,0,0)
        bin_state[0] = (state['tree']['top'] - 200) // 50
        tree_dist = state['tree']['dist']

        if tree_dist < 0:
            bin_state[1] = 0:
        elif tree_dist < 50:
            bin_state[1] = 1:
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
        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

iters = 300
learner = Learner()

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=100,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    # Reset the state of the learner.
    learner.reset()



    
