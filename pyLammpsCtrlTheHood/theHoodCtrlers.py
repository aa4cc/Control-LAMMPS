import numpy as np
from pyLammpsCtrlTheHood import theHoodFunc as theHood

###############################
######## Actuators ############
###############################        

class SpringAFM:
    # Representation of the AFM control as a spring that is attached to a group of atoms.
    
    def __init__(self, MD_p_ctrl, MD_p_setup_sys, lmp) :
        # ctrl_atoms_IDs: IDs of the atoms that will be connected to the spring based on a region
        # X0: fixed side of the spring

        # Get positions and sort them based on their IDs
        X = lmp.gather_atoms("x",1,3) # All positions
        IDs = np.array(lmp.gather_atoms("id",0,1)) # All IDs
        sort_idx = np.argsort(IDs)
        X[0::3] = np.array(X[0::3])[sort_idx]
        X[1::3] = np.array(X[1::3])[sort_idx]
        X[2::3] = np.array(X[2::3])[sort_idx]
        
        # Get Masses of atoms
        Types = np.array(lmp.gather_atoms("type",0,1)) # All IDs
        mass = lmp.extract_atom("mass")[1:3] # TODO: make it generalized
        Masses = np.zeros(len(IDs))
        for i in range(0, len(IDs)):
            if Types[i] == 1:
                Masses[i] = mass[0]
            else:
                Masses[i] = mass[1]
        Masses = Masses[sort_idx]
        
        # Fixed variables
        self.ctrl_atoms_IDs = theHood.extract_IDs_ctrl_region(MD_p_setup_sys, MD_p_ctrl["ctrl_boundary_arr"], lmp) # IDs (1,..., N) of the atoms that are    
        self.K = MD_p_ctrl["spring_stiffness"]  # Spring stiffness
        self.X0 = theHood.get_COM_IDs_atoms(self.ctrl_atoms_IDs, X, Masses) # Point of the end of the spring attached to a COM of group of atoms
        self.R0 = MD_p_ctrl["spring_offset_R0"] # Equilibrium elongation of the spring 
        self.Ts = MD_p_ctrl["Ts"]               # Control (not internal LAMMPS Ts) period of the simulation
        self.initX1 = self.X0 + np.array([1, 0, 0])*(MD_p_ctrl["spring_offset_R0"] + MD_p_ctrl["init_elongation"])  
        # Variables changing in time
        self.V = MD_p_ctrl["init_spring_speed"]
        self.axes = MD_p_ctrl["axes"]
        self.X1 = self.initX1  # Free-end of the spring
        # self.elong = MD_p_ctrl["init_elongation"]
    
    def lmpFix_command(self, V, traj, step, lmp):
        # Propagate (in time) the free-end to the next position based on the speed vector "V = [Vx, Vy, Vz]"
        # Set the speed of the free-end for the next iteration
        # "traj" and "step" allows to set different behavior for different trajectories or current time.
        
        self.X1[self.axes] = self.X1[self.axes] + V*self.Ts
        # self.elong = self.X1 - theHood.get_COM_IDs_atoms(self.ctrl_atoms_IDs, X)
        
        # Take a current instance of the lmp (lammps handler) and create a "fix" to control the system: https://docs.lammps.org/fix_spring.html
        lmp.command("fix spring_ctrl ctrl_group spring tether " + str(self.K) + " " + " ".join(map(str,self.X1)) + " " + str(self.R0))   
        
        return self.X1
                               
    def reset(self, lmp):
        # This function should be called at every start of the lammps simulation when the system is controlled through the spring 
        
        # Create a group given by the IDs of the atoms that will be 
        lmp.command("group ctrl_group id " + " ".join(map(str,self.ctrl_atoms_IDs)))
        self.X1 = self.initX1 # Reinitialize the free-end position to an initial point
        
        
###############################
######## Controllers ##########
###############################              
        
class OpenLoopCtrler:
    def __init__(self, Ts, Vref) :
        self.Ts = Ts
        self.Vref = Vref
        
    def ctrl(self, Y, traj, ii, comm):
        return self.Vref[:,ii-1]
    
    def reset(self):
        None

class PID:
    # Assumes that we control X,Y,Z coordinates
    def __init__(self, K, U_bounds, sigma_noise, Ts, Yref) :
        # Yref: reference in position only
        self.Kp = K[0:3,:]
        self.Ki = K[3:6,:]
        self.Kd = K[6:9,:]
        self.U_bounds = U_bounds
        self.sigma = sigma_noise
        self.Err_prev = np.zeros(3)
        self.Err_pprev = np.zeros(3)
        self.U_prev = np.zeros(3)
        self.Ts = Ts
        self.curr_traj = -1
        self.Yref = Yref
        
    def ctrl(self, Y, traj, ii, comm):
        # NOTE: Allow parallelization
        # TODO: do not let ii < 0
        # Y: measurement from the system
        # Output: the action, is velocity of the AFM in three directions XYZ
        
        Pos = Y[0::2] # Take only positions
        
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        
        if nproc > 1:
            if rank == 0:
                Err = self.Yref[:,ii-1] - Pos
                U_1 = +(np.ones(3) + self.Ki[:,traj] + self.Kd[:,traj])*Err
                U_2 = -(np.ones(3) + 2*self.Kd[:,traj])*self.Err_prev
                U_3 = +self.Kd[:,traj]*self.Err_pprev
                U_rnd = np.random.multivariate_normal([0,0,0], self.sigma)
                
                U = self.U_prev + self.Kp[:,traj]*(U_1 + U_2 + U_3) + U_rnd
                self.Err_pprev = self.Err_prev
                self.Err_prev = Err
                self.U_prev = U
            else:
                U = None
                
            U = comm.bcast(U, root=0)
        else:
            Err = self.Yref[:,ii-1] - Pos
            U_1 = +(np.ones(3) + self.Ki[:,traj] + self.Kd[:,traj])*Err
            U_2 = -(np.ones(3) + 2*self.Kd[:,traj])*self.Err_prev
            U_3 = +self.Kd[:,traj]*self.Err_pprev
            U_rnd = np.random.multivariate_normal([0,0,0], self.sigma)
            
            U = self.U_prev + self.Kp[:,traj]*(U_1 + U_2 + U_3) + U_rnd
            self.Err_pprev = self.Err_prev
            self.Err_prev = Err
            self.U_prev = U
        
        return bndV(U, self.U_bounds[:,1], self.U_bounds[:,0])
    
    def reset(self):
        # Reset to zero
        self.Err_pprev = self.Err_pprev*0
        self.Err_prev = self.Err_prev*0
        self.U_prev = self.U_prev*0
               
###############################        
######## Aux functions ########
###############################        
        
def bndV(val, max, min):
    # Bound value
    return np.maximum(np.minimum(val,max),min)
