# A guide to the RangL environment (in OpenAI Gym)

This folder contains a copy of the RangL environment which will be used for modeling the NBM output data. It can be used right away to train an agent. However to gain an advantage in the competition, you can consider training your agent in a modified environment (although evaluation will still use this original environment!). 

The RangL environment is designed to be easily modified and has the following modular structure:

```Dependencies```  

```Helpers```  

```Environment class```

Any modifications should be made to the Helpers (updating the dependencies as needed). 

## Helpers 

The Helper classes are:

* Parameters -- contains all challenge-specific parameters
* State -- contains all state information and provides the following methods:
    * initialise_state
    * reset
    * to_observation -- returns just the observations which will be passed to the agent   
    * is_done -- checks whether the episode has finished

and the Helper functions are:

* observation_space -- specifies the agent's observation space
* action_space -- specifies the agent's action space
* apply_action -- applies an action to the state and calculates the reward
* verify_constraints -- checks whether the actions have violated any pre-specified constraints
* randomise -- adds random noise to the state (representing uncertainty over the future) 
* record -- records data for graphing at the end of the episode
* plot_episode -- plots the recorded data
* score -- returns score for the full episode

## Environment class

The environment class provides the standard initialise, reset, step and score methods required by OpenAI Gym. It also provides these additional methods:

* seed -- allows the random seed to be specified
* plot -- applies the plot_episode function 

## Random seeding

Explicit seeds persist upon reset() . So:

*	If you set env.seed(None), reset() produces different noise
*	If you set e.g. env.seed(42), reset() produces identical noise

