# Adaptive Focusing
This is intended to be a minimum working example demonstrating fully adaptive and partially adaptive focusing on simulated passive sonar data

## Pre-requisites
Requires PyTorch, Numpy and Scipy

## Running the example
To run the simulation example, run:
`python adaptive_focusing_example.py simulation_configs/example_simulation.json`
This will generate simulation data according to the parameters in `example_simulation.json`, run both fully adaptive focusing and partially adaptive focusing, and then beamform using a wideband MVDR beamformer and plot the results.
This is intended to be a toy example demonstrating how to use the code, and so, we don't expect much different results from pre-steering, which is showed in the plots for comparison.