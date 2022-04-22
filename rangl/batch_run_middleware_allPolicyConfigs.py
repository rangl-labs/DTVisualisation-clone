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


for i in np.array(['H','M','L']):
    for j in np.array(['H','M','L']):
        for k in np.array(['H','M','L']):
            run_middleware(np.array([i,j,k]))