## Changes

Changed how Q and a are stored. Now representing them as a dictionary. The keys are hash values for each state 'tuple'. Essentially I call state_tupler from a function called state_hash. Then, state_hash takes the return value of state_tupler and converts it into a hash. Therefore, each state has a unique has obtained by simply calling state_hash(state).

Began implementing TD-value learning which is a hybrid model-based and model-free method (it's mentioned in the course notes). However, right now it doesn't seem to be working.

### Images

* scores.png - the default output when running program. Has whatever was run last.

* scores1.png - First successful run of q-learn. Q and a represented as multi-dimensional np.arrays. Discount value is 1.0. 

* scores2.png - Same as scores1.png except axes are labeled

* dictionaryuse.png - the result of changing the base q-learn code from using np.arrays for Q and a to dictionary. Clearly, using dictionaries did not break the code 