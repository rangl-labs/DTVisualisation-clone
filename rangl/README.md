# Middleware and RangL environment

The middleware converts policy configurations of (thermal comfort, mass electrification, energy resilience) to fixed action used to evolve the RangL environment forward in time, which is a simulation of year-by-year deployments/installations of the 4 technlogies (Heat pumps, Solar PV, External wall Insulation, EV charge points).

The middleware then maps the buildings in NBM data to 3 visualization levels: UK, Manchester, Local street, and calculates the relevant metrics and averages year-by-year, and finally outputs the calculations to plain .csv files.

## The RangL environment

RangL uses the [Openai Gym framework](https://gym.openai.com). To install the RangL environment on your local machine, 

1. If necessary, install the pip package manager (you can do this by running the `get-pip.py` Python script)

2. Run `pip install -e .`

The `reference_environment` folder contains the environment used in this middleware. To modify it for development purposes, look at its `README` file.

## Running the middleware

The (modified) environments can be tested by running the middleware using `test_middleware.py`, which is modularized as a function in `middleware_func.py`.

The middleware can be run both in batch and interactively. The `batch_run_middleware_allPolicyConfigs.py` will loop through all 3x3 = 27 possible policy configurations and feed each of the policy config into the `middleware_func.py`, whereas the `run_middleware_policyInputByUser.py` allows for the user to input a policy config and generates the outputs accordingly.
