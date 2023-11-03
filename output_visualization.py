import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from pyLammpsCtrlTheHood import theHoodObsrv as observs
from lammps import lammps

pickle_file = "24-10-2023_18-52-15_curr_dataset"
file1 = open("./sim_data/" + pickle_file + ".pkl", 'rb')
dataset = pickle.load(file1)

fig = plt.figure() 
X = dataset["X"]
U = dataset["U"]
Pos_ref = dataset["Pos_ref"]
X_top_layer = dataset["M_top_layer"]["X"]
MD_p_ctrl = dataset["MD_p_ctrl"]
obsrvIDs = dataset["obsrvIDs"]
Ntraj = MD_p_ctrl["Ntraj"]
num_timesteps = int(np.size(X,1)/Ntraj)
time = list(range(0, num_timesteps))

for i in range(0,MD_p_ctrl["obsv_N_reg"]):
    plt.plot(time, X[6*i, :])
plt.plot(time, Pos_ref.T[:,0])    
plt.show()
    
plt.plot(time, U[0,:])
plt.show()

time = np.linspace(0, num_timesteps*Ntraj-1, num_timesteps*Ntraj)
for i in range(0,len(obsrvIDs)):
    plt.plot(time, X_top_layer[6*i,:], 1)
plt.show()