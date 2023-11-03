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

# How to run the examples

Two examples are provided:

1) **sim1_prepare_system.py**
2) **sim2_closed_loop_ctrl.py**


The examples can be run direcly using python but for faster simulation, we recommend using **mpirun**

An example og calling:
- ` mpirun -np $N /bin/python3 ./sim1_prepare_system.py `

where **N** is number of CPU cores used for execution.