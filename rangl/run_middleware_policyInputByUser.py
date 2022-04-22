import logging
from pathlib import Path

import pandas as pd
import numpy as np
import gym
# import reference_environment
from middleware_func import run_middleware

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

policy_configs = np.array(['L','L','L']) # ['thermal comfort', 'mass electrification', 'energy resilience']
policy_configs[0] = input("Enter the policy config for Thermal Comfort (input 'L' for low, 'M' for medium, 'H' for high): ")
policy_configs[1] = input("Enter the policy config for Mass Electrification (input 'L' for low, 'M' for medium, 'H' for high): ")
policy_configs[2] = input("Enter the policy config for Energy Resilience (input 'L' for low, 'M' for medium, 'H' for high): ")

run_middleware(policy_configs)