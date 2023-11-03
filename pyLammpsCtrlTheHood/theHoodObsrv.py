import numpy as np
from pyLammpsCtrlTheHood import theHoodFunc as theHood


# COM: center-of-mass

class FkObsrv:
    # Observe positions and velocities of COMs associated with get_COM_IDs_atoms
    # regions of the top layer
    
    def __init__(self, MD_p_ctrl, Pos, IDs, Masses, axes=[0,1,2]) :
        # IDs: IDs of all atoms in the dataset
        # Fixed variables
        self.Ts = MD_p_ctrl["Ts"] 
        self.N = MD_p_ctrl["obsv_N_reg"]       # Divide the observable region into N parts in X direction, each part will be represented by a COM
        self.obsvr_reg = MD_p_ctrl["obsv_reg"] # The whole observable region: [X_bounds, Y_bounds, Z_bounds]
        order = MD_p_ctrl["order"]             # "desc": for region being sorted from highest X coordinate to lowest, "asc": the opposite
        self.axes = axes
        self.partsIDs = init_particle_regions_ID(Pos, IDs, self.N, self.obsvr_reg, order)
        self.COMs_Pos = None
        self.Masses = Masses
        
        sort_idx = np.argsort(IDs)
        # Resorted based on IDs
        x = np.array(Pos[0::3])[sort_idx]
        y = np.array(Pos[1::3])[sort_idx]
        z = np.array(Pos[2::3])[sort_idx]
        Pos[0::3] = x
        Pos[1::3] = y
        Pos[2::3] = z
        COMs_Pos_curr = np.zeros((self.N,3))
        for i in range(0, self.N):
            COMs_Pos_curr[i,:] = theHood.get_COM_IDs_atoms(self.partsIDs[i], Pos, Masses)
        # Reduce observable axes
        COMs_Pos_curr = COMs_Pos_curr[:, axes]
        
        self.Pos_COM_init = COMs_Pos_curr # Constant variable (to store initial COMs)
        
        # Variables changing in time
        self.COMs_Pos = COMs_Pos_curr     # Previous COMS of subsystems.        
        
    
    def measure(self, X, lmp=None):
        # Take the full-state vector X of the system and TODO: the auxilary measured variables (temperature, energy, etc.) 
        # and return only the variables defined by the observer
        
        # Function for using in a control loop
        
        # The X vector should be already sorted by the IDs of the atoms (ascending) and be in a form: 
        #   [pos_x1, vel_x1, pos_y1, vel_y1, pos_z1, vel_z1, pos_x2, vel_x2, pos_y2, vel_y2, pos_z2, vel_z2, ....]
        
        # TODO: incorporate aux_X
        
        Pos = X[0::2]
        COMs_Pos_curr = np.zeros((self.N,3))
        for i in range(0, self.N):
            COMs_Pos_curr[i,:] = theHood.get_COM_IDs_atoms(self.partsIDs[i], Pos, self.Masses)
        # Reduce observable axes
        COMs_Pos_curr = COMs_Pos_curr[:, self.axes]        

        V = (COMs_Pos_curr - self.COMs_Pos)/self.Ts

        # Update
        self.COMs_Pos = COMs_Pos_curr
        
        X_out = np.zeros((self.N*len(self.axes)*2))
        X_out[0::2] = np.reshape(COMs_Pos_curr, int(self.N*len(self.axes)))
        X_out[1::2] = np.reshape(V, int(self.N*len(self.axes)))
        return X_out
              
    def observTrajN(self, Y_in, U_in, Ntraj, Nsim):
        # NOTE: Function assumes that all trajectories start in the same state (position)
        # This function serves for converting state measurements (pos, vel) of individual atoms into measurements of FK-model-like particles 
        # Observe Ntraj trajectories
        # Y_in: 
        
        nx = self.N*2*len(self.axes)
        M = np.zeros((nx, Ntraj*Nsim)) # *2 for velocities *3 for XYZ coordinates
        Mp = np.zeros((nx, Ntraj*Nsim))
    
        for traj in range(0, Ntraj):
            self.reset()
            idx_start = (traj)*Nsim 
            idx_end = (traj+1)*Nsim
            # M_in_curr_traj = M[:,idx_start+traj:idx_end+traj, Nsim+1]
            Y_in_curr_traj = Y_in[:, idx_start:idx_end]
            M_curr_traj = np.zeros((nx, Nsim+1))
                
            # Init
            M_curr_traj[0::2, 0] = self.COMs_Pos.flatten()       
            for t in range(1, Nsim+1):
                M_curr_traj[:, t] = self.measure(Y_in_curr_traj[:, t-1])
            
            M[:, idx_start:idx_end] = M_curr_traj[:,0:-1]
            Mp[:, idx_start:idx_end] = M_curr_traj[:,1:]
        U = U_in[self.axes, :]
        return M, Mp, U
    
    def reset(self):
        # This function should be called at every start of the lammps simulation when the system is controlled through the spring 
        self.COMs_Pos = self.Pos_COM_init



class RegionObsrv:
    # Observe the states of atoms from a given region
    def __init__(self,MD_p_ctrl,comm,Pos):
        self.Ts = MD_p_ctrl["Ts"] 
        self.N = MD_p_ctrl["obsv_N_reg"]       # Divide the observable region into N parts in X direction, each part will be represented by a COM
        self.obsvr_reg = MD_p_ctrl["obsv_reg"] # The whole observable region: [X_bounds, Y_bounds, Z_bounds] 
        self.obsrvIDs = obsv_region_IDs(self.obsvr_reg, Pos.flatten())
        
        
    def measure(self, X, lmp=None):
        # Assume that X is sorted by IDs
        Pos = X[0::2]
        Vel = X[1::2]
        
        x = np.array(Pos[0::3])[self.obsrvIDs-1]
        y = np.array(Pos[1::3])[self.obsrvIDs-1] 
        z = np.array(Pos[2::3])[self.obsrvIDs-1]
        
        vx = np.array(Vel[0::3])[self.obsrvIDs-1]
        vy = np.array(Vel[0::3])[self.obsrvIDs-1]
        vz = np.array(Vel[0::3])[self.obsrvIDs-1]
        
        X_measur = np.zeros((len(self.obsrvIDs)*6))
        X_measur[0::6] = x
        X_measur[1::6] = vx
        X_measur[2::6] = y
        X_measur[3::6] = vy
        X_measur[4::6] = z
        X_measur[5::6] = vz
        
        return X_measur
        
        
        
###############################        
######## Aux functions ########
###############################        
        
        
def obsv_region_IDs(obsvr_reg, Pos):
    # Assumptions: Pos are sorted by IDs
    x = np.array(Pos[0::3])
    y = np.array(Pos[1::3]) 
    z = np.array(Pos[2::3])
    
    indices_region = np.where( (x>=obsvr_reg[0][0]) & (x<=obsvr_reg[0][1]) & 
                        (y>=obsvr_reg[1][0]) & (y<=obsvr_reg[1][1]) &
                        (z>=obsvr_reg[2][0]) & (z<=obsvr_reg[2][1]))
    IDs = indices_region[0]+1 # Take IDs of atoms only from the observable region (e.g. top layer)
    return IDs
            
        
def init_particle_regions_ID(Pos, IDs, N, obsvr_reg, order):
    # Get all positions
    # Pos = lmp.gather_atoms("x",1,3) 
    # IDs = np.array(lmp.gather_atoms("id",0,1)) # All IDs
    # num_atoms = lmp.extract_global("natoms")
    x = np.array(Pos[0::3])
    y = np.array(Pos[1::3]) 
    z = np.array(Pos[2::3])
    
    # Sort by IDS
    sort_idx = np.argsort(IDs)

    # Resorted based on IDs
    IDs = IDs[sort_idx]
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]
    # Pos[0::3] = x
    # Pos[1::3] = y
    # Pos[2::3] = z
    
    # Get atoms that are in the observable region
    indices_region = np.where( (x>=obsvr_reg[0][0]) & (x<=obsvr_reg[0][1]) & 
                          (y>=obsvr_reg[1][0]) & (y<=obsvr_reg[1][1]) &
                          (z>=obsvr_reg[2][0]) & (z<=obsvr_reg[2][1]))
    IDs = IDs[indices_region] # Take atoms only from the observable region (e.g. top layer)
    x = x[indices_region]
    y = y[indices_region]
    z = z[indices_region]
    
    # get minimum and maximum in X and Y directions
    XY_bounds = [[np.min(x), np.max(x)], [np.min(y), np.max(y)]]
    
    # Convert the region into a list of N 1D-array: each 1D array will contain IDs corresponding to the k-th part.
    # NOTE and TODO: Working only for division along X direction
    parts = [None] * N
    for i in range(0,N):
        inverval_len_X = (XY_bounds[0][1] - XY_bounds[0][0])/N
        curr_X_bound = XY_bounds[0][0] + [inverval_len_X*i, inverval_len_X*(i+1)]
        indices_region_part = np.where( (x>=curr_X_bound[0]) & (x<=curr_X_bound[1]) )
        if order == "asc":
            parts[i] = IDs[indices_region_part]
        else:
            parts[N - i - 1] = IDs[indices_region_part]            
    return parts
        
        
def bndV(val, max, min):
    # Bound value
    return np.maximum(np.minimum(val,max),min)
