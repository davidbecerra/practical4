# Practical 4: Reinforcement Learning 
#### David Becerra, Conrad Shock, Wentao Xu

### Files

* q-learn.py : Implementation of Q-learn with initial/original discretization.

* q-learn-new-disc.py: Implementation of Q-learn with refined discretization (best learner algorithm).

* td-value.py : Implementation of td-value with refined discretization.

### Usage
To run the desired file (q-learn.py, q-learn-new-disc.py, or td-value.py) type the following command in a terminal:
`python <file-name>`

The command will open the game GUI, and the learner will start to play the game. All the learners run for 300 epochs except for q-learn-new-disc.py which runs for 150 epochs. When the learners complete, a plot of the scores over all the epochs is displayed. To close the program, exit out of the final plot, or exit out of the GUI while it is still running. 