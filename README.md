# OPF-NN-Learning-Framework (SWR-26-081)

A deep learning framework for AC Optimal Power Flow (AC-OPF) that integrates a differentiable optimization layer (PyomoOptLayer) into a neural network to produce nearly feasible and optimal power dispatch solutions. 

## Overview

A neural network is used to obtain a partial set of AC-OPF solutions, including generator active power and voltages.

In the testing stage, either the optimization model or a PF solver can be used to obtain the complete AC-OPF solutions.

## License

This repository is distributed under the BSD 3-Clause License. 

## Acknowledgements

More details of the proposed framework can be found in the work titled ``A hard-constrained NN learning framework for rapidly restoring AC-OPF from DC-OPF".

## Requirements

- Python 
- PyTorch
- NumPy
- SciPy
- Pyomo
- pyomolayers (recommended to install first)
- pypower (used for ``makeYbus", option if the Ybus is passed)
## Required Inputs

The data input

| Variable | Description | Unit |
|----------|-------------|------|
| `Pd` | Active load demand | MW |
| `Qd` | Reactive load demand | MVAR |
| `Pg_DC` | Active generator output from the DC-OPF model | MW |
| `Qg` | Reactive generator output | MVAR | (option for training)
| `Vm` | Bus voltage magnitudes | p.u. | (option for training)
| `Va` | Bus voltage angles | degrees | (option for training)

Power grid topology and line parameters

such as `case_ACTIVSg2000.mat` that can be obtained from Matpower

## Architecture

The `NNSolver` neural network has the following structure:
- Input: `[Pd, Qd, Pg_DC]` concatenated (dim = `nbus * 2 + ng`)
- Fully connected NN
- Output: predicted `[pg_all (all generator (PV) buses), vg (generator buses and the reference bus )]` (dim = `2 * ng -1`)
  - Voltage magnitudes are passed through Tanh and `ACOPFProblem.complete_voltage` to satisfy bounds [Vmin, Vmax]

## Training
The training loop:
1. Run the optimization layer on NN predictions, compute projection loss, update NN weights.
2. Replay buffer stored (input, primal solution) pairs for additional supervised training.


