import numpy as np
import numpy.matlib
from lammps import lammps, LAMMPS_INT, LMP_STYLE_GLOBAL
import math    
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


def get_COM_IDs_atoms(select_IDs, Pos, Masses):
    # Compute Center-of-mass (xyz position) of atoms given by their IDs
    
    idxes = (np.array(select_IDs)-1).astype(int) # IDs are indexed from 1,..., N
    # Assume, that positions are already sorted
    x = np.array(Pos[0::3])[idxes] 
    y = np.array(Pos[1::3])[idxes] 
    z = np.array(Pos[2::3])[idxes]
    
    sumMass = sum(Masses[idxes])
    
    x_com = 0
    y_com = 0
    z_com = 0
    for i in range(0,len(idxes)):
        m = Masses[idxes[i]]
        x_com = x_com + m*x[i]/sumMass
        y_com = y_com + m*y[i]/sumMass
        z_com = z_com + m*z[i]/sumMass

    return np.array([x_com, y_com, z_com])


def init_sys(MD_p_glob, init_data_file_dir, lmp):
    lammps_init_cmds = [
    ### simulation parameters definition
    "clear", # Just cleaning the Lammps memory
    "units           " + MD_p_glob["units"],
    "boundary        " + MD_p_glob["boundary"], # TODO: Maybe needed?
    ### force fields definition
    "atom_style      " + MD_p_glob["atom_style"],
    "pair_style      " + MD_p_glob["pair_style"],
    "read_data       " + init_data_file_dir,
    MD_p_glob["force_field_descr"][0],
    MD_p_glob["force_field_descr"][1],
    MD_p_glob["force_field_descr"][2],
    "neighbor        3.0 bin"
    ]
    
    lmp.commands_list(lammps_init_cmds)


def thermal_equilibration(MD_p_glob, MD_p_therm, lmp):
    # Init the system
    init_sys(MD_p_glob, MD_p_therm["md_sys_in_dir"], lmp)
    
    # Thermal equilibration
    s_temp = str(MD_p_therm["temp"])
    steps = int(MD_p_therm["temp_equi_time"]/MD_p_glob["dtime"])
    lammps_therm_equi_commands = [
        "variable        steps_dump equal 100  # steps; 1 conf every 10 fs",
        "variable        steps_thermo equal 1  # steps; 1 line every 0.1 fs",
        ### groups definition
        "region r_S1  block INF INF INF INF 0.0 2.0", 
        "region r_Mo1 block INF INF INF INF 2.0 4.0", 
        "region r_S2  block INF INF INF INF 4.0 5.0",
        "region r_S3  block INF INF INF INF 7.0 8.2", 
        "region r_Mo2 block INF INF INF INF 8.7 9.7", 
        "region r_S4  block INF INF INF INF 10  11.5",
        "group Mo1 region r_Mo1", 
        "group Mo2 region r_Mo2", 
        "group S1  region r_S1",
        "group S2  region r_S2", 
        "group S3  region r_S3", 
        "group S4  region r_S4", 
        "group bottom union Mo1 S1 S2", 
        "group top    union Mo2 S3 S4", 
        "group mobile union Mo1 Mo2 S2 S3 S4",
        ### fixes for motion and load
        "fix             f2 all nvt temp " + s_temp + " " + s_temp + " $(100.0*dt)",
        ### initialize velocities
        "velocity        all create " + s_temp + " " + str(MD_p_therm["seed"]),
        "velocity        all zero linear",
        ### thermo definition
        "thermo          ${steps_thermo}",
        "thermo_style    custom time etotal pe ke temp",
        ### timestep
        "timestep        " + str(MD_p_glob["dtime"]),
        ### dumps
        "dump		    xyz all custom ${steps_dump} " + MD_p_therm["md_sys_traj_dir"] + " element xu yu zu",
        "dump_modify 	xyz element Mo S",
        "dump_modify 	xyz sort id",
        ### run
        "run            " + str(steps),
        ### write data
        "write_data " + MD_p_therm["md_sys_data_dir"]
    ]
    lmp.commands_list(lammps_therm_equi_commands)
    print("------------------------------")
    print("Structure of the system theramlly equilibrated at the given temperation. The system is written into a file.")
    print("------------------------------")


def minimize_pot_energy(MD_p_glob, MD_p_min_En, lmp) :
    # Init the system
    init_sys(MD_p_glob, MD_p_min_En["md_sys_in_dir"], lmp)
    
    # Minimize the potential energy
    lammps_init_cmd_equilibrate = [
        ### dumps
        "dump		    xyz all custom 100 " + MD_p_min_En["md_sys_minim_en_traj_dir"] + " element xu yu zu id fx fy fz",
        "dump_modify 	xyz element Mo S",
        "dump_modify 	xyz sort id",
        ### start minimization 
        "thermo          1",
        "thermo_style    custom time lx ly xy pe fmax fnorm",
        "fix             f1 all box/relax x 0.0 y 0.0 xy 0.0 couple xy nreset 1 fixedpoint 0.0 0.0 0.0",
        "min_style       cg",
        "minimize        0.0 1.0e-10 1000000 1000000",
        ### write data
        "write_data " + MD_p_min_En["md_sys_minim_en_data_dir"],
    ]
    lmp.commands_list(lammps_init_cmd_equilibrate)
    print("------------------------------")
    print("Structure of the system optimized to get a structure with a locally minimal potential energy. The system is written into a file.")
    print("------------------------------")


def create_system(MD_p_glob, MD_p_setup_sys, lmp):
    init_sys(MD_p_glob, MD_p_setup_sys["material_file"], lmp)

    #########################
    #### Create a sample ####
    #########################
    create_sample(MD_p_setup_sys["replicate_arr"], MD_p_glob, MD_p_setup_sys["x_frac"], MD_p_setup_sys["y_frac"],lmp)
    lmp.command("write_data " + MD_p_setup_sys["md_sys_replicated_dir"])
    print("------------------------------")
    print("System created. The structure is written into a file.")
    print("------------------------------")
    
    
def extract_IDs_ctrl_region(MD_p_setup_sys, extract_arr, lmp):
    one_sheet_num_atoms = 3 # there are three atoms in one sheet of the MoS2
    num_atoms = lmp.extract_global("natoms")
    replicate_arr = MD_p_setup_sys["replicate_arr"]
    top_sheet_num_atoms = num_atoms - replicate_arr[0]*replicate_arr[1]*one_sheet_num_atoms
    top_sheet_layer_num_atms = int(top_sheet_num_atoms/3) # There are three layers in the top sheet
    
    y_top_sheet_rows = int(MD_p_setup_sys["replicate_arr"][1]*MD_p_setup_sys["y_frac"])
    x_top_sheet_num = int(top_sheet_layer_num_atms/y_top_sheet_rows)
    
    X = lmp.gather_atoms("x",1,3) # All positions
    IDs = np.array(lmp.gather_atoms("id",0,1)) # All IDs
    x = np.array(X[0::3])
    y = np.array(X[1::3])
    z = np.array(X[2::3])


    sort_idx_z = np.argsort(z)
    idx_top_sheet_z_sort = sort_idx_z[-top_sheet_num_atoms:] # Take only indices from the top layer (hopefully)
    
    x_top_sheet = x[idx_top_sheet_z_sort]
    y_top_sheet = y[idx_top_sheet_z_sort]
    IDs_top_sheet = IDs[idx_top_sheet_z_sort]
    
    IDs_extracted = []
    num_layers = 3 # only in the top sheet
    for i in range(num_layers-extract_arr[2],num_layers): # Controlled will always be only the atoms from the top sheet. 
        idx_start_layer = int(top_sheet_layer_num_atms*i)
        x_curr_layer = x_top_sheet[idx_start_layer:idx_start_layer + top_sheet_layer_num_atms]
        y_curr_layer = y_top_sheet[idx_start_layer:idx_start_layer + top_sheet_layer_num_atms]
        IDs_curr_layer = IDs_top_sheet[idx_start_layer:idx_start_layer + top_sheet_layer_num_atms]

        sort_idx_y = np.argsort(y_curr_layer)
        for j in range(0,y_top_sheet_rows):
            sort_idx_y = np.argsort(y_curr_layer)
            sorted_y_IDs = IDs_curr_layer[sort_idx_y]
            
            idx_start_line = int(x_top_sheet_num*j)
            x_curr_line = x_curr_layer[sort_idx_y][idx_start_line:idx_start_line+x_top_sheet_num]
            x_sort_idx = np.argsort(x_curr_line)
            IDs_curr_line = sorted_y_IDs[idx_start_line:idx_start_line+x_top_sheet_num]
            IDs_extracted.extend(IDs_curr_line[x_sort_idx][-extract_arr[0]:])
    return IDs_extracted
   
    
def create_sample(replicate_arr, MD_params, x_frac, y_frac,lmp):   
    # Currently only for MoS2
    # Replication in Z direaction is not implemented
    
    num_layers = 6

    # Replicate and delete in X direction
    replicate_system(np.array([replicate_arr[0], 1,1]), lmp)
    ids = np.array(lmp.gather_atoms("id",0,1))
    X = lmp.gather_atoms("x",1,3)
    x = np.array(X[0::3])
    z = np.array(X[2::3])
    z_mean = np.mean(z)
    
    sort_idx_z = np.argsort(z)
    cutoff_len_x = round(replicate_arr[0]*x_frac)
    
    ids_to_remove_x = []
    for i in range(3,num_layers): # Remove atoms only from the top sheet 
        # There are "replicate_arr[0]" atoms in each layer. Remove a portion given by x_frac, sort first
        idx_start_layer = replicate_arr[0]*i
        curr_x = x[sort_idx_z[idx_start_layer:idx_start_layer+replicate_arr[0]]]
        sort_idx_x = np.argsort(curr_x)
        
        sorted_z_based_x = sort_idx_z[sort_idx_x + i*replicate_arr[0]] 
        ids_to_remove_x.extend(ids[sorted_z_based_x[cutoff_len_x:]])
    lmp.command("group rmIDx id " + " ".join(map(str,ids_to_remove_x)))
    lmp.command("delete_atoms group rmIDx")
    
    num_atoms_single_line = lmp.extract_global("natoms")
    # # Replicate in Y direction
    replicate_system(np.array([1, replicate_arr[1],1]), lmp)
    num_atoms = lmp.extract_global("natoms")
    ids = np.array(lmp.gather_atoms("id",0,1))
    X = lmp.gather_atoms("x",1,3)
    y = np.array(X[1::3])
    z = np.array(X[2::3])
    
    # Round to nearest multiple of the "num_atoms_single_line"
    num_rm_y_atoms = round(round(num_atoms*y_frac)/num_atoms_single_line)*num_atoms_single_line
    sort_idx_y = np.argsort(y)[num_rm_y_atoms:]
    
    sort_idx_z = np.nonzero(z > z_mean)[0]
    inters_idx = np.intersect1d(sort_idx_z, sort_idx_y)
    ids_to_remove_y = ids[inters_idx]
    if len(ids_to_remove_y) != 0:
        lmp.command("group rmIDy id " + " ".join(map(str,ids_to_remove_y)))
        lmp.command("delete_atoms group rmIDy")
    

def replicate_system(replicate_arr, lmp):
    # Commands for LAMMPS to replicate the system based on the given array replicate_arr
    lmp.command("replicate " + " ".join(map(str,replicate_arr)))


def X_V_to_state(X,V, num_atoms):
    curr_state = np.zeros(6*num_atoms)
    for jj in range(0, num_atoms):
        idxs = list(range(0+jj*6,6+jj*6))
        curr_state[idxs] = np.array([X[0+jj*3], V[0+jj*3], X[1+jj*3], V[1+jj*3], X[2+jj*3], V[2+jj*3]])
    return curr_state


def generate_traj_spring(start_pos, speed_vec, MD_p_ctrl):
    # TODO: make this function more parametrizable: non-constant speed
    # generate_ref_traj_xAtoms_temp might be already it but there is the Temp which we do not want.
    # speed_vec = MD_p_ctrl["spring_speed"]
    Tsim = MD_p_ctrl["Tsim"]
    Ts = MD_p_ctrl["Ts"]
    time = np.linspace(0,Tsim, num=int(Tsim/Ts)+1)
    traj = start_pos.reshape((3,1)) + np.array(speed_vec).reshape((3,1))*time
    
    return traj


def generate_ref_traj_X(Nsim, Ts, Pos0, v_ref, idx_stop):
# Generate ref trajectory for x-coordinates of atoms, Position and speed
    # Pos0: offset for each particle
    # v_ref: reference speed for each particle in XYZ direction
    # idx_stop: index of changing from reference speed to zero
    
    n_ax = np.size(v_ref) # Number of dimensions/axis (XYZ)
    num_particles = int(np.size(Pos0)/n_ax)
    Tsim = Nsim*Ts
    t_ref = np.linspace(0,Tsim-Ts, int(Tsim/Ts))
    
    # Reference for one atom in "nx" coordinates
    pos_ref = np.zeros(([n_ax, int(Nsim)]))
    vel_ref = np.zeros(([n_ax, int(Nsim)]))
    for i in range(0, n_ax):
        vel_ref[i,:] = v_ref[i]*np.ones(np.size(t_ref))
        pos_ref[i,:] = v_ref[i]*t_ref
        pos_ref[i,idx_stop:] = pos_ref[i,idx_stop]
        vel_ref[i,idx_stop:] = 0 
    # Replicate for all particles
    Pos_ref = np.matlib.repmat(pos_ref, num_particles, 1)
    Vel_ref = np.matlib.repmat(vel_ref, num_particles, 1)
    
    # Add offset
    for i in range(0, int(len(Pos0))):
        Pos_ref[n_ax*i:n_ax*(i+1), :] = Pos_ref[i, :] + Pos0[i,:].reshape(n_ax,1)
    
    # TempRef = tempRef*np.ones((1,Nsim))
    # X_ref = np.append(X_ref, TempRef, axis=0 ) # Just try to keep constant temperature
    X_ref = np.zeros((Pos_ref.shape[0] + Vel_ref.shape[0],Nsim))
    X_ref[0::2, :] = Pos_ref
    X_ref[1::2, :] = Vel_ref
    return X_ref, Pos_ref, Vel_ref


def extract_wrapped_pos_and_vel(num_atoms, id_fixID, posU_fixID, comm, lmp):
    # NOTE: Allows parallelization
    # Output positions and velocities are sorted by ID
    # Positions are wrapped (periodic w.r.t. the simulation box)
        
    # rank = comm.Get_rank()
    # nproc = comm.Get_size() 
    
    Pos = np.array(lmp.gather_atoms("x", 1, 3)).reshape((num_atoms,3))
    Vel = np.array(lmp.gather_atoms("v", 1, 3)).reshape((num_atoms,3))
    id_fix = np.array(lmp.gather_atoms("id",0,1))
    sort_idx = np.argsort(id_fix)
    Pos = Pos[sort_idx]
    Vel = Vel[sort_idx] 
    
    # TODO: unwrap the wrapped positions, something like this:
    # image = np.array(lmp.gather_atoms("image", 1, 1))
    # if max(image) > 0.0001:
    #     print("Image is higher")
    
    return Pos.flatten(), Vel.flatten()


def extract_unwrapped_pos_and_vel(num_atoms, id_fixID, posU_fixID, comm, lmp):
    # NOTE: Allows parallelization
    # - Each processor works only with a subset of atoms. 
    # Output positions and velocities are sorted by ID
    # Positions are unwrapped (not periodic w.r.t. the simulation box)
        
    # TODO: Not functioning properly
        
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    
    # Init memory
    Pos_u = np.reshape(np.empty(num_atoms*3, dtype='d'), (num_atoms, 3)) # Init memory
    
    #### Gather speeds:
    Vel_u = np.array(lmp.gather_atoms("v", 1, 3)).reshape((num_atoms,3))
    
    if nproc > 1:
        ####### Parallelization #######
        # Get valid IDs
        IDs = np.array(lmp.extract_fix(id_fixID,1,1)[0:num_atoms]) # "extract_fix" allows to get position of atoms computed by the single processor
        IDs = IDs[IDs !=0].astype(int)
        num_valid = len(IDs)
        
        # Positions from each processor
        Pos_u_frac = np.reshape(np.array(lmp.extract_fix(posU_fixID,1,2).contents[0:num_valid*3]), (num_valid, 3))

        # Sending positions and IDs of each Rank to the root=0
        if rank != 0:
            sendbuf = {
                'IDs' : IDs,
                'data' : Pos_u_frac
            }
            comm.send(sendbuf, dest=0)
        else: # Root (rank = 0)
            IDroot = np.empty(num_atoms, dtype=int)
            IDroot[IDs-1] = IDs # insert Root-computed IDs, ID are indexed from 1
            Pos_u[IDs-1, :] = Pos_u_frac # insert Root-computed positions
            
            ### Insert data from other processes
            for i in range(1, nproc):
                idata = comm.recv(source=i)    
                IDroot[idata["IDs"]-1] = idata["IDs"]
                Pos_u[idata["IDs"]-1, :] = idata["data"]
    else: 
        ####### No parallelization #######
        id_fix = np.array(lmp.gather_atoms("id",0,1))
        sort_idx = np.argsort(id_fix)
        Pos_u  = np.reshape(np.array(lmp.extract_fix(posU_fixID,1,2).contents[0:num_atoms*3]), (num_atoms, 3))
        Pos_u = Pos_u[sort_idx]
        Vel_u = Vel_u[sort_idx] 
    
    return Pos_u.flatten(), Vel_u.flatten()


def extract_IDs_region(lmp, MD_param_struct, region):
    # Deprecated, 13.7.2023
    num_atoms = MD_param_struct["num_atoms"]
    pos = extract_unwrapped_pos_and_vel(num_atoms, MD_param_struct["fix_ID"], MD_param_struct["fix_unwr_XV"], lmp, num_atoms)[0]
    pos_x = pos[0:3*num_atoms:3]
    pos_y = pos[1:3*num_atoms:3]
    pos_z = pos[2:3*num_atoms:3]
    id_region = []
    for i in range(0,num_atoms):
        if region["x"][0] <= pos_x[i] < region["x"][1]:
            if region["y"][0] <= pos_y[i] < region["y"][1]:
                if region["z"][0] <= pos_z[i] < region["z"][1]:
                    id_region.append(i+1) # +1 because LAMMPS is indexing from 1 to num_atoms
    
    return np.array(id_region)


def collectData_lammps(actuator, sensor, ctrler, MD_p_ctrl, MD_p_glob, ctrl_cmds, comm, lmp):
    # This function allows collecting multiple trajectories with different control in each trajectory.
    
    Ts = MD_p_ctrl["Ts"]
    Tsim = MD_p_ctrl["Tsim"]
    steps = Ts/MD_p_ctrl["Ts_intern"]
    Ntraj = MD_p_ctrl["Ntraj"]
    Nsim = int(Tsim/Ts) # Number of control periods
    num_atoms = lmp.extract_global("natoms")
    num_inpt = MD_p_ctrl["num_inpts"]
    num_outp = MD_p_ctrl["num_outputs"]
    
    # Pre-allocate arrays for full state
    X_full = np.zeros((6*num_atoms,Ntraj*Nsim))
    Y_full = np.zeros((6*num_atoms,Ntraj*Nsim))
    
    # Pre-allocate arrays for measurements
    X = np.zeros((num_outp, Ntraj*Nsim))
    Y = np.zeros((num_outp, Ntraj*Nsim))
    U = np.zeros((num_inpt,Ntraj*Nsim))
    
    # Pre-allocate arrays for auxilary (extended state) variables
    PotE_X = np.zeros(Ntraj*Nsim)
    KinE_X = np.zeros(Ntraj*Nsim)
    Temp_X = np.zeros(Ntraj*Nsim)
    PotE_Y = np.zeros(Ntraj*Nsim)
    KinE_Y = np.zeros(Ntraj*Nsim)
    Temp_Y = np.zeros(Ntraj*Nsim)
    Actuator_out = np.zeros((num_inpt,Ntraj*Nsim))
    
    rank = comm.Get_rank()
    
    ### For every trajectory ###
    for traj in range(0, Ntraj):
        if rank == 0:
            print("---------------------------------------------------")
            print("Trajectory %s out of %s"%(str(traj+1), str(Ntraj)))
            print("---------------------------------------------------")
            
        X_traj_obsrv = np.zeros((num_outp, Nsim+1))
        
        U_traj = np.zeros((num_inpt,Nsim))
        PotE_traj = np.zeros(Nsim+1)
        KinE_traj = np.zeros(Nsim+1)
        Temp_traj = np.zeros(Nsim+1)
        actuator_out = np.zeros((num_inpt, Nsim))

        lmp = lammps() # TODO: Currently, the simulation needs to be reinitilized for every trajectory. Some type of "reset" could speed up the simulations        
        init_sys(MD_p_glob, MD_p_ctrl["md_sys_in_dir"], lmp) # Init the system by loading result of the thermal equilibration
        lmp.commands_list(ctrl_cmds)
        lmp.command("run 1") # Get the first iteration
         
        actuator.reset(lmp)        
        ctrler.reset()   
        sensor.reset()
        
        # Initial state, sorted by ID
        Pos_u_init, Vel_u_init = extract_wrapped_pos_and_vel(num_atoms, MD_p_ctrl["fix_ID"], MD_p_ctrl["fix_unwr_XV"], comm, lmp)
        X_traj_full = X_V_to_state(Pos_u_init,Vel_u_init, num_atoms)
        X_traj_obsrv[:,0] = sensor.measure(X_traj_full, lmp) # Should also return auxilary variables (temp, kin energy, ...)        
        
        PotE_traj[0] = np.mean(lmp.extract_variable("pe", "all", 1))
        KinE_traj[0] = np.mean(lmp.extract_variable("ke", "all", 1))
        Temp_traj[0] = lmp.extract_compute("thermo_temp", LMP_STYLE_GLOBAL, LAMMPS_INT)
                
        ### For every timestep in one trajectory ###
        for ii in range(1, Nsim+1):      
            # - Controller computes velocity (from position diff)
            curr_V = ctrler.ctrl(X_traj_obsrv[:,ii-1], traj, ii, comm)
            # - Spring changes its free-end position (from given velocity)
            actuator_out[:,ii-1] = actuator.lmpFix_command(curr_V, traj, ii, lmp)
            lmp.command("run " + str(int(steps)))

            ## For the next iteration, Extract
            Pos_u, Vel_u = extract_wrapped_pos_and_vel(num_atoms, MD_p_ctrl["fix_ID"], MD_p_ctrl["fix_unwr_XV"], comm, lmp)        
            X_traj_full = X_V_to_state(Pos_u,Vel_u, num_atoms)
            
            U_traj[:,ii-1] = curr_V
            PotE_traj[ii] = np.mean(lmp.extract_variable("pe", "all", 1))
            KinE_traj[ii] = np.mean(lmp.extract_variable("ke", "all", 1))
            Temp_traj[ii] = lmp.extract_compute("thermo_temp", LMP_STYLE_GLOBAL, LAMMPS_INT)
            X_traj_obsrv[:,ii] = sensor.measure(X_traj_full, lmp)
            
            
        #### Read trajectory from a file ####
        X_traj_full_file = np.zeros((6*num_atoms, Nsim+1))
        dump_file = open(MD_p_ctrl["md_ctrl_sys_traj_dir"], 'r')
        for i in range(0,9): dump_file.readline() # Skip the first header
        # For every step in the trajectory
        for curr_timestep in range(0, Nsim+1):
            Pos_all = np.zeros((num_atoms*3))
            Vel_all = np.zeros((num_atoms*3))
            for i in range(0, num_atoms):
                line = dump_file.readline()
                num_line = np.fromstring(line, dtype=float, sep=' ')
                Pos_all[3*i:3*(i+1)] = num_line[1:4]
                Vel_all[3*i:3*(i+1)] = num_line[4:7]
            for i in range(0,9): dump_file.readline()
            X_traj_full_file[:, curr_timestep] = X_V_to_state(Pos_all,Vel_all, num_atoms)
        dump_file.close()
        #####################################
        
        idx_start = (traj)*Nsim
        idx_end = (traj+1)*Nsim
        X_full[:,idx_start:idx_end] = X_traj_full_file[:,0:-1]
        Y_full[:,idx_start:idx_end] = X_traj_full_file[:,1:]
        
        Actuator_out[:,idx_start:idx_end] = actuator_out
        
        PotE_X[idx_start:idx_end] = PotE_traj[0:-1]
        KinE_X[idx_start:idx_end] = KinE_traj[0:-1]
        Temp_X[idx_start:idx_end] = Temp_traj[0:-1]
        
        PotE_Y[idx_start:idx_end] = PotE_traj[1:]
        KinE_Y[idx_start:idx_end] = KinE_traj[1:]
        Temp_Y[idx_start:idx_end] = Temp_traj[1:]
    
        # TODO: HERE, save the measurements
        X[:,idx_start:idx_end] = X_traj_obsrv[:,0:-1]
        Y[:,idx_start:idx_end] = X_traj_obsrv[:,1:]
        U[:,idx_start:idx_end] = U_traj
    
        aux_measur = {
            "PotE_X" : PotE_X,
            "KinE_X" : KinE_X,
            "Temp_X" : Temp_X,
            "PotE_Y" : PotE_Y,
            "KinE_Y" : KinE_Y,
            "Temp_Y" : Temp_Y,
            "Actuator_out" : Actuator_out,
        }
        
        full_state = {
            "X" : X_full,
            "Y" : Y_full
        }
        
    return X, Y, U, full_state, aux_measur, lmp


def collectMeasurementIDs(M, IDs):
    # Take a measurement M (the whole trajectory) and select only those that correspond to the IDs
    # Each column corresponds to a time instance, the data are in rows.
    M_new = np.zeros((len(IDs)*6, M.shape[1]))    
    for i in range(0,len(IDs)):
        M_new[6*i:6*i+6,:] = M[6*(IDs[i]-1):6*(IDs[i]-1)+6,:] # -1 to convert ID to python index
    return M_new


def group_state_idxs(group_idxs, stateSize):
    # Create list of indices corresponding to the atom group indices 
    Idxs = np.zeros(len(group_idxs)*stateSize)
    
    for ii in range(0, len(group_idxs)):
        Idxs[ii*stateSize:(ii+1)*stateSize] = group_idxs[ii]*stateSize+np.array(range(0,stateSize))
    return Idxs


## Visualization fuctions

def animate_MD(fig, X, num_atoms, quant, Nsim, lims):
    Xx = X[0:6*num_atoms:6][:]
    Xy = X[2:6*num_atoms:6][:]
    Xz = X[4:6*num_atoms:6][:]
        
    anim_frames = int(math.floor(Nsim/quant))
    # fig = plt.figure()    

    def update_points(num, x, y, z, points):
        txt.set_text('Time={:d} Femtoseconds'.format(num*quant)) # for debug purposes

        # calculate the new sets of coordinates here. The resulting arrays should have the same shape
        # as the original x,y,z
        new_x = Xx[:,num*quant]
        new_y = Xy[:,num*quant]
        new_z = Xz[:,num*quant]

        # update properties
        points.set_data(new_x,new_y)
        points.set_3d_properties(new_z, 'z')

        # return modified artists
        return points,txt    

    xlims = lims["xlims"]
    ylims = lims["ylims"]
    zlims = lims["zlims"]
    
    ax = p3.Axes3D(fig)
    ax.view_init(azim= -135)
    ax.set_xlim3d(xlims[0], xlims[1])
    ax.set_ylim3d(ylims[0], ylims[1])
    ax.set_zlim3d(zlims[0], zlims[1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel('z')   
    
    x=Xx[0:num_atoms, 0]
    y=Xy[0:num_atoms, 0]
    z=Xz[0:num_atoms, 0]

    points, = ax.plot(x, y, z, '*')
    txt = fig.suptitle('')
    

    ani=animation.FuncAnimation(fig, update_points, frames=anim_frames, fargs=(x, y, z, points), interval = 1)

    return plt, ani


def mColors(idx):
    colors = [ '#1f77b4', '#ff7f0e','#2ca02c', '#d62728',
               '#9467bd', '#8c564b','#e377c2', '#7f7f7f',
               '#bcbd22', '#17becf']
    return colors[idx]
    