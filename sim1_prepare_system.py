# import sys
# import os
# from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from lammps import lammps
import numpy as np  
from pyLammpsCtrlTheHood import theHoodFunc as theHood
 
# For faster simulation, run this script with multiple CPU cores with a command:
# mpirun -np $NCORES $PYTHON_PATH ./sim1_whole_pipeline_prepare_system.py
# ... where 
# ... ... NCORES is number of available cores, e.g.,    8
# ... ... PYTHON_PATH is python path, e.g.,             /bin/python3


# General description of the system
MD_p_glob = {
    "curr_dir_name"     : "./",
    "md_files_dir"      : "md_data",    # Relative path of the directory with files that describe the input Molecular system
    "sim_data_dir"      : "sim_data/",  # Relative path of the directory that stores all simulation data".
    "dtime"             : 1,            # integration step of the simulation in femtoseconds
    "units"             : "real",       # Setting units of the physical variable, see "units" in LAMMPS manual
    "boundary"          : "p p p",      # boundary conditions in all directions
    "atom_style"        : "atomic",
    "pair_style"        : "hybrid/overlay sw/mod lj/cut 10.0",
    "force_field_descr" : [
        "pair_coeff      * * sw/mod ./md_data/mos2.sw Mo S", 
        "pair_coeff      * * lj/cut 0.0     0.0",
        "pair_coeff      2 2 lj/cut 0.15981 3.13",
    ]
}

# Parameters for creating - replication (setup) of the system (stationary simumlation):
MD_p_setup_sys = {
    "material_file" : "./md_data/9007660.data", # The "unit cell" description, view vith "vesta 9007660.data"
    "replicate_arr" : np.array([40, 15, 1]),    # Replicate the unit system to XYZ directions
    "x_frac"        : 0.4 , # Cutoff of the top layer in X
    "y_frac"        : 1 , # Cutoff of the top layer in Y
    "md_sys_replicated_dir"    : MD_p_glob["curr_dir_name"] +  MD_p_glob["sim_data_dir"] + "sys_replicated.data",
}

# Parameters for minimizing the potential energy of the structure:
MD_p_min_En = {
    "md_sys_in_dir" : MD_p_setup_sys["md_sys_replicated_dir"],
    "md_sys_minim_en_data_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_minim_energ.data",
    "md_sys_minim_en_traj_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_minim_energ.lammpstrj"
}

# Parameters for thermal equilibration:
MD_p_therm = {     
    "md_sys_in_dir" : MD_p_min_En["md_sys_minim_en_data_dir"],          
    "temp" : 300,
    "seed" : 450,
    "temp_equi_time" : 1*1000, # femtoseconds, For real application, this number should be at least 50*1000. i.e., 50 ps
    "md_sys_traj_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_therm_equi.lammpstrj",
    "md_sys_data_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_therm_equi.data",
}

# Initialization of the LAMMPS object
lmp = lammps()

##################################################
#### Whole process of creating a system       ####
##################################################
# - All individual steps can be run independently, provided that the previous step was done because the next step relies on the output of the previous step

##################################################
#### Creating a system                        #### 
##################################################
# - Take a description of the material (https://next-gen.materialsproject.org/) and replicate to create a bigger system
# - Output is written into a file that is used in the function "theHood.minimize_pot_energy"
theHood.create_system(MD_p_glob, MD_p_setup_sys, lmp)


##################################################
#### Minimize the energy of the system        ####
##################################################
# - The structure of the system needs to be optimize to get a configuration with (locally) minimal potential energy
# - This step takes a while
# - Output is written into a file that is used in the function "theHood.thermal_equilibration"
theHood.minimize_pot_energy(MD_p_glob, MD_p_min_En, lmp)


##################################################
#### Thermal equilibration 
##################################################
# - Equilibrate the system at the set temperature so the speeds of the atoms are sampled from the Maxwell distribution
# - The duration of the equilibration should be long, at least several pikoseconds
theHood.thermal_equilibration(MD_p_glob, MD_p_therm, lmp)


# view the final structure using "vmd" by calling a command: "vmd *.lammpstrj"
