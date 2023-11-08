# Control-LAMMPS
This repository contains a LAMMPS interface for feedback control of molecular dynamics.


# Installation
The process was tested on a desktop computer with Linux OS (Ubuntu 22.04.3 LTS, jammy)

## Dependencies

For correct installation, the recommended steps are:

1) Install Python 3 or newer
2) Install MPI4PY library for Python
3) Download and build latest version of LAMMPS 
    - Building flags: **DBUILD_SHARED_LIBS=yes -DPKG_MANYBODY=yes**
        - The command:
            - ` cmake ../cmake -DBUILD_SHARED_LIBS=yes -DPKG_MANYBODY=yes `
4) Install Python 3 virtual enviroment (required for correct link between python MPI4PY library and LAMMPS)
5) Install python wrapper for LAMMPS

Several other standard Python libraries are also required.

# How to run the examples

Two examples are provided:

1) **sim1_prepare_system.py**
2) **sim2_closed_loop_ctrl.py**


The examples can be run direcly using python but for faster simulation, we recommend using **mpirun**

An example of running the example in parallel:
- ` mpirun -np $N /bin/python3 ./sim1_prepare_system.py `

where **N** is number of CPU cores used for execution.

# Visualization

A dataset of a simulation with PID control is available in **./sim_data/sim_data.rar**.
The data is compressed into a .rar to reduce the data size.
You can then use script **output_visualization.py** to plot the **.pkl** data or VMD (https://www.ks.uiuc.edu/Research/vmd/) to visualize LAMMPS trajectory file **.lammpstrj** .
