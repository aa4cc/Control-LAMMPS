from mpi4py import MPI
import math
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np  
import os
from lammps import lammps
from pyLammpsCtrlTheHood import theHoodCtrlers as ctrlers
from pyLammpsCtrlTheHood import theHoodFunc as theHood
from pyLammpsCtrlTheHood import theHoodObsrv as observs


# Init parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Current processor
nprocs = comm.Get_size() # Total number of processors

if nprocs > 1:
    if rank == 0:
        now = datetime.now()
    else:
        now = None
    now = comm.bcast(now, root=0)
else:
    now = datetime.now()
    
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

MD_p_glob = {
    "curr_dir_name"     : "./",
    "md_files_dir"      : "md_data",
    "sim_data_dir"      : "sim_data/", # relative path to the "curr_dir_name".
    "dtime"             : 1, # integration step of the simulation in femtoseconds
    "units"             : "real", # see the lammps manual https://docs.lammps.org/units.html
    "boundary"          : "p p p", # boundary conditions in all directions
    "atom_style"        : "atomic",
    "pair_style"        : "hybrid/overlay sw/mod lj/cut 10.0",
    "force_field_descr" : [
        "pair_coeff      * * sw/mod ./md_data/mos2.sw Mo S", 
        "pair_coeff      * * lj/cut 0.0     0.0",
        "pair_coeff      2 2 lj/cut 0.15981 3.13",
    ],
}

# Parameters for creating - replication (setup) of the system (stationary sim)
MD_p_setup_sys = {
    "material_file" : "./md_data/9007660.data", # The "unit system" description, view vith "vesta 9007660.data"
    "replicate_arr" : np.array([40, 15, 1]),    # Replicate the unit system to XYZ directions
    "x_frac"        : 0.4 , # Cutoff of the top layer in X
    "y_frac"        : 1 , # Cutoff of the top layer in Y, NOTE: replicate_arr[1]*y_frac should be integer!
    "md_sys_replicated_dir"    : MD_p_glob["curr_dir_name"] +  MD_p_glob["sim_data_dir"] + "sys_replicated.data",
}

# Parameters for minimizing the energy of the structure (stationary sim)
MD_p_min_En = {
    "md_sys_in_dir" : MD_p_setup_sys["md_sys_replicated_dir"],
    "md_sys_minim_en_data_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_minim_energ.data",
    "md_sys_minim_en_traj_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_minim_energ.lammpstrj"
}

# Parameters of thermal equilibration:
MD_p_therm = {     
    "md_sys_in_dir" : MD_p_min_En["md_sys_minim_en_data_dir"],          
    "temp" : 300,
    "seed" : 666,
    "temp_equi_time" : 0.5*100, # femtoseconds, For real application, this number should be at least 50*1000
    "md_sys_traj_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_therm_equi.lammpstrj",
    "md_sys_data_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + "sys_therm_equi.data",
}

MD_p_ctrl = {
    "md_sys_in_dir" : MD_p_therm["md_sys_data_dir"], # In this step, take the output of thermal equilibration as an input
    "fix_ID" : "fix_ID",
    "fix_unwr_XV" : "fix_unwr_XV",
    "Ts_intern" : 15, # in femtoseconds: integration step in Lammps
    "Ts" : 50000,      # Control period"   
    "md_ctrl_sys_traj_dir" : MD_p_glob["curr_dir_name"] + MD_p_glob["sim_data_dir"] + dt_string + "_spring_ctrl_sys.lammpstrj",
    "ctrl_boundary_arr" : [3,-1,3], # Define the ctrl region:  5 in X direction, all atoms in Y direction, all atoms (always three) in Z direction
    "spring_stiffness" : 80/0.695101, # Convert N/m --to-- kcal/(mol*Angstrom)/Angstrom, In literature, k=10 N/m is usually used
    "spring_offset_R0" : 0.0, # See R0 in lammps DOC spring R0
    "init_spring_speed" : np.array([0,0,0])*1.0e-5, # convert m/s to Ang/femtosec
    "max_speed" : np.array([1,0,0])*1.0e-5,
    "Ntraj" : 1,            # Number of trajectories. i.e., simulation runs.
    "Tsim" : 200000,        # Simulation time in femtoseconds
    "num_inpts" : 3,        # Number of real inputs. In our case: velocity in XYZ direction
    "gamma" : 50,           # Dissipation coefficient as described by the Langevin thermostat, [unit] = ps^-1 
    "temp" : 300,           # Temperature of the system controlled by the thermostat
    "init_elongation" : 6,  # Initial elongation of the spring that models the AFM
    "axes" : [0,1,2],       # 
    "end_pos" : [50,0,0],   # End position for the control
    # Observer parameters:
    "obsv_reg" : [[-math.inf, math.inf],[-math.inf, math.inf],[6, 15]],     # Region of the system that is observable
    "obsv_N_reg" : 1,    # The whole observable region is represented by one Center-of-mass
    "num_outputs" : 1*6, # Number of measured variables: The state of each Center-of-mass have 3 position and 3 velocity variables 
    "order" : "desc", # So the subsystem at index 0 will have highest X coordinate
}

steps = MD_p_ctrl["Ts"]/MD_p_ctrl["Ts_intern"]

lammps_cmds_ctrl = [
    ### Defining groups of atoms so they can be references during the simulation
    "region r_S1   block INF INF INF INF 0.0 2.0",      
    "region r_Mo1  block INF INF INF INF 2.0 4.0",      
    "region r_S2   block INF INF INF INF 4.0 5.0",      
    "region r_S3   block INF INF INF INF 7.0 8.2",
    "region r_Mo2  block INF INF INF INF 8.2 9.7",
    "region r_S4   block INF INF INF INF 10  11.5",
    "group Mo1 region r_Mo1",
    "group Mo2 region r_Mo2",
    "group S1  region r_S1",
    "group S2  region r_S2",
    "group S3  region r_S3",
    "group S4  region r_S4",
    "group           bottom union Mo1 S1 S2",
    "group           top    union Mo2 S3 S4",
    "group           mobile union Mo1 Mo2 S2 S3 S4",
    ### "fixes" for motion and load
    "fix 1 mobile langevin "+str(MD_p_ctrl["temp"])+" "+str(MD_p_ctrl["temp"])+" "+str(MD_p_ctrl["gamma"])+" 14545", # https://docs.lammps.org/fix_langevin.html
    "fix             f0  all    store/force",
    "fix             f2a mobile nve",
    "fix             " + MD_p_ctrl["fix_unwr_XV"] + " all store/state 1 xu yu zu",
    "fix             " + MD_p_ctrl["fix_ID"] + " all store/state 1 id",
    #
    "velocity        S1 set 0.0 0.0 0.0",       # Bottom layer is fixed.
    "velocity        S4 zero angular",
    "compute 1 all pe/atom",
    "compute 2 all ke/atom",
    "variable ke atom ke",   # So we can extract it via **lmp.extract_variable** command   
    "variable pe atom pe",    
    ### thermo definition
    "thermo          1",
    "thermo_style    custom time etotal temp",
    ### timestep
    "timestep " + str(MD_p_ctrl["Ts_intern"]), # in fs
    
    ### dumps
    "dump		xyz all custom " + str(int(steps)) + " " + MD_p_ctrl["md_ctrl_sys_traj_dir"] + " element xu yu zu vx vy vz",
    "dump_modify 	xyz element 1 2",
    "dump_modify 	xyz sort id",
]

## NOTE: This script requires the file "*.data" that describes a system, e.g., sys_therm_equi.data
if not os.path.exists(MD_p_ctrl["md_sys_in_dir"]):
    print("ERROR: The init file for LAMMPS does not exists or the path is wrong!")
    exit(1)


# Initialize LAMMPS
lmp = lammps()

# Initialize the system
theHood.init_sys(MD_p_glob, MD_p_ctrl["md_sys_in_dir"], lmp) # Init the system by loading result of the thermal equilibration and defined force field

# Get Masses of atoms
IDs = np.array(lmp.gather_atoms("id",0,1)) # All IDs
Types = np.array(lmp.gather_atoms("type",0,1)) # All IDs
mass = lmp.extract_atom("mass")[1:3] #
Masses = np.zeros(len(IDs))
for i in range(0, len(IDs)):
    if Types[i] == 1:
        Masses[i] = mass[0]
    else:
        Masses[i] = mass[1]

# Create the actuator: spring-based SpringAFM
springAFM = ctrlers.SpringAFM(MD_p_ctrl, MD_p_setup_sys, lmp)

# Create observer and initialize it 
Pos = np.array(lmp.gather_atoms("x", 1, 3)).reshape((len(IDs),3))[np.argsort(IDs)]
FK_obsrv = observs.FkObsrv(MD_p_ctrl, Pos.flatten(), IDs, Masses)

# Create PID controller and setting the PID constants
K = np.zeros((3*3,MD_p_ctrl["Ntraj"]))
for i in range (0, MD_p_ctrl["Ntraj"]):
    K[0:3,i] = [0.00000019,0,0]  # Px, Py, Pz
    K[3:6,i] = [0.00015,0,0]      # Ix, Iy, Iz
    K[6:9,i] = [0.01,0,0]      # Dx, Dy, Dz, 
U_bounds = np.array((-MD_p_ctrl["max_speed"], MD_p_ctrl["max_speed"])).T # Bounds on speed in X, Y, Z
sigma_noise = np.diag([0, 0, 0]) # Allowing to add noise to the control input defined by covariance matrix of a white-noise

# Create and provide the reference trajectory to the Controller. The PID takes only the position reference: Pos_ref 
Nsim = int(MD_p_ctrl["Tsim"]/MD_p_ctrl["Ts"])
_ , Pos_ref, _ = theHood.generate_ref_traj_X(Nsim, MD_p_ctrl["Ts"], FK_obsrv.COMs_Pos + np.array([MD_p_ctrl["end_pos"]]), MD_p_ctrl["init_spring_speed"], int(Nsim*3/5))

ctrler = ctrlers.PID(K, U_bounds, sigma_noise, MD_p_ctrl["Ts"], Pos_ref)

# Run and gather data
X, Y, U, full_M , aux_X, lmp = theHood.collectData_lammps(springAFM, FK_obsrv, ctrler, MD_p_ctrl, MD_p_glob, lammps_cmds_ctrl,
                                                                                                          comm, lmp)

# Extract states of only the top layer
regObsrv = observs.RegionObsrv(MD_p_ctrl,comm,Pos)
num_atoms_top = len(regObsrv.obsrvIDs)

fig = plt.figure() 
time = list(range(0,np.size(X,1)))
if rank == 0:
    # Save only the states of top layer
    X_top_layer = np.zeros((num_atoms_top*6, np.size(X,1)))
    Y_top_layer = np.zeros((num_atoms_top*6, np.size(X,1)))
    # For every step of the simulation
    for i in range(0, np.size(X,1)):
        X_top_layer[:,i] = regObsrv.measure(full_M["X"][:,i], lmp)
        Y_top_layer[:,i] = regObsrv.measure(full_M["Y"][:,i], lmp)
    M_top_layer = {
        "X" : X_top_layer,
        "Y" : Y_top_layer
    }
    
    dataset = {
    "X" : X, "Y" : Y, "U" : U,
    "aux_observ" : aux_X,
    "M_top_layer" : M_top_layer,
    "Pos_ref" : Pos_ref,
    "MD_p_ctrl" : MD_p_ctrl,
    "obsrvIDs" : regObsrv.obsrvIDs
    }
    file_name = MD_p_glob["curr_dir_name"] +  MD_p_glob["sim_data_dir"] + dt_string + '_curr_dataset.pkl'
    file = open(file_name, 'wb')
    pickle.dump(dataset, file)


MPI.Finalize()