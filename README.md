Here is a clean README.md for your thesis GitHub repository. You can copy and paste it directly.

# FPGA-Based Hardware Acceleration of a Learning-Based Control Workflow

This repository contains the implementation files and experiment scripts for my thesis project on FPGA-based hardware acceleration of a learning-based control workflow. The work focuses on implementing and evaluating a Deep Q-Network based control workflow for an inverted pendulum application on an FPGA platform.

The project uses High-Level Synthesis to implement the main learning workflow as a hardware accelerator and uses PYNQ on the processing system to control the accelerator, run experiments, and collect results.

## Project Overview

Learning-based control methods can require a large number of repeated computations during training, validation, and evaluation. When these workflows are executed only in software, the total execution time can become high, especially when multiple experiments or random seeds are required.

This project studies how such a workflow can be mapped to FPGA hardware. The main focus is not only on inference, but on the complete workflow, including:

- Training
- Validation
- Final evaluation
- Timing measurement
- Cycle estimation
- Performance analysis

The inverted pendulum is used as the control benchmark because it is a nonlinear and unstable system that requires continuous decision-making.

## Main Features

- FPGA implementation of a DQN-based learning workflow
- Inverted pendulum control environment
- Training, validation, and evaluation execution on hardware
- PYNQ-based control from Python
- Multi-seed experiment support
- Timing and cycle measurement
- Internal neural-network profiling
- Performance metric calculation from stable-upright steps
- Batch-size comparison experiments

## Target Platform

The implementation is designed for the following FPGA platform:

Board  : ZCU104
Device : Zynq UltraScale+ MPSoC
Flow   : Vitis HLS / Vivado / PYNQ
