# Adaptive Focusing
This is intended to be a minimum working example demonstrating fully adaptive and partially adaptive focusing on simulated passive sonar data

## Pre-requisites
Requires PyTorch, Numpy and Matplotlib

## Running the example
To run the simulation example, run:

`python adaptive_focusing_example.py simulation_configs/example_simulation.json`

This will generate simulation data according to the parameters in `example_simulation.json`, run both fully adaptive focusing and partially adaptive focusing, and then beamform using a wideband MVDR beamformer and plot the results.
This is intended to be a toy example demonstrating how to use the code, and so, we don't expect much different results from pre-steering, which is showed in the plots for comparison.

## Coding conventions

### Variable names
This code follows a naming convention where scalars have no trailing underscores, 1D vectors have a single trailing underscore, 2D vectors two trailing underscores, and so on. This allows for easy looping without a jumble of variable names.

For example:

```
a = 10 # scalar
a_ = np.arange(10) # 1D vector
A__ = np.arange([10,10]) # 2D matrix
A___ = np.arange([10,10,10]) # 3D array (or "Tensor")
```

This allows easy looping:

```
for a_ in A__:
  print(a_.shape)
```

### Batch matrix multiplications
This code heavily utilizes PyTorch Tensor batch matrix multiplication which are then automatically paralellized when GPU hardware is specified.