#!/usr/bin/python3
import numpy as np
import bisect as bi
from mpl_toolkits import mplot3d
import matplotlib.pyplot as pt
import control as ct
from control.matlab import *
pt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "text.latex.preamble": "\\usepackage{biolinum}", "mathtext.fontset": "dejavusans", 'figure.constrained_layout.use': True})
pt.close()
fig, ax = pt.subplots(3,3)

# Helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def edInt(a,s,b):
    return np.linspace(a,b,int(np.floor((b-a)/s)))

# Used for poorman's nearest-neighbour regular-grid interpolation
def idx(aList, val):
    i = bi.bisect(aList, val) # right
    if i <= 0: # first
        pass
    elif len(aList) == i: # last
        i -= 1
    elif val - aList[i-1] < aList[i] - val: # left
        i -= 1
    return i

def lookup(aList, vals):
    """result is undefined if vals is not sorted"""
    indices = []
    v = vals
    if type(v) in [int,float,np.int64,np.float64]:
        v = [v]
    for k in np.sort(v):
        indices.append(idx(aList,k))
    return np.array(indices, dtype=np.uint32)

# Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = ...; infStValInc = ...; nullCtr = 0
T = ... # time horizon
U = ... # admissible controls
x1, x2 = np.meshgrid(...) # admissible states
a1 = a2 = .01
b1 = b2 = .005
A = np.array([[-a1, b1], [b2, -a2]])
B = np.array([[1, 0], [0, 0]])

# System constants
Q = ...
R = ...
z0 = ... # initial state (only for simulation)
w = ... # reference (or set-point)
N = T.size
V = np.zeros((x2.shape[0], x1.shape[1], N)) # optimal cost
u_opt = V.copy() # optimal control
V[:,:,N-1] = Q[0,0] * x1**2 + Q[1,1] * x2**2 # terminal penalties (goal-state = 0)
err_x1 = np.zeros((x2.shape[0], x1.shape[1]))
err_x2 = err_x1.copy()
depth=0 # for value snapshots

# Algorithm (operating on isolated equilibrium) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k in range(N-1,depth,-1):
    for i in range(0,x2.shape[0]):
        for j in range(0,x1.shape[1]):
            x = np.array([x1[i,j], x2[i,j]])
            L = ...
            # forward Euler
            new_x1 = ...
            new_x2 = ...
            i_new_x1 = lookup(x1[0,:],new_x1) # naive interpolation
            i_new_x2 = lookup(x2[:,0],new_x2)
            i_U = range(0,U.size)
            err_x1[i,j] = max(new_x1) # truncated dynamics
            err_x2[i,j] = new_x2
            # make sure admissible state/control exists
            if (i_new_x1.size != 0 and i_new_x2.size != 0):
                i_U, V[i,j,k-1] = ...
            else:
                i_U, V[i,j,k-1] = lookup(U,nullCtr)[0][0], V[i,j,k] + infStValInc
            u_opt[i,j,k-1] = U[i_U] # i_U should be a singleton list

# Simulation (from z0 using u_opt(x)) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z = np.zeros((T.size,2))
z[0,:] = z0
u = np.zeros((T.size,1))
for k in range(0,T.size-1):
    if not(z[k,0]-w[0]>=x1[0,0] and z[k,0]-w[0]<=x1[0,-1] # admissibility check
           and z[k,1]-w[1]>=x2[0,0] and z[k,1]-w[1]<=x2[-1,0]): 
        print(f'stopped at {k}/{T.size}', z[k,0], z[k,1])
        break
    u[k] = u_opt[lookup(x2[:,0],z[k,1]-w[1]), lookup(x1[0,:],z[k,0]-w[0]), depth]
    z[k+1,:] = ...
    # TODO: (b) second controller (you can duplicate data structures)
    # TODO: (c) jumps

# Visualisation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vz0 = V[lookup(x2[:,0],z0[1]-w[1]), lookup(x1[0,:],z0[0]-w[0]), 0]
inval_x1 = 100 * np.where(np.logical_or(err_x1[:,:] > x1[1,-1], err_x1[:,:] < x1[1,1]))[0].size \
    / (x1[:,1].size * x2[1,:].size)
inval_x2 = 100 * np.where(np.logical_or(err_x2[:,:] > x2[-1,1], err_x2[:,:] < x2[1,1]))[0].size \
    / (x1[:,1].size * x2[1,:].size)
           
ax[0,0].plot(T, z[:,0], 'g-', label=r'$x_1$')
ax[0,0].set(xlabel=r't [s]', ylabel=r'$x_1$')
ax[0,0].axhline(y=w[0], color='r', linestyle='--')
ax[1,0].plot(T, z[:,1], 'g-', label=r'$x_2$')
ax[1,0].set(xlabel=r't [s]', ylabel=r'$x_2$')
ax[1,0].axhline(y=w[1], color='r', linestyle='--')
ax[2,0].plot(T, u, 'g-', label=r'$u^*$')
ax[2,0].set(xlabel=r't [s]', ylabel=r'$u^*$')

ax1 = fig.add_subplot(3,3,2, projection='3d')
ax1.plot_surface(x2, x1, V[:,:,1])
ax1.set(xlabel=r'$x_2$', ylabel=r'$x_1$', zlabel=r'$V(x(t),t)$', title=f'$V(x({depth}),{depth})$')
ax1.plot(z0[1]-w[1], z0[0]-w[0], Vz0, marker='*')

kp = N-1
ax2 = fig.add_subplot(3,3,5, projection='3d')
ax2.plot_surface(x2, x1, V[:,:,kp])
ax2.set(xlabel=r'$x_2$', ylabel=r'$x_1$', zlabel=r'$V(x(t),t)$', title=f'V(x(.),{kp})')

ax3 = fig.add_subplot(3,3,8, projection='3d')
ax3.plot_surface(x2, x1, u_opt[:,:,depth])
ax3.set(xlabel=r'$x_2$', ylabel=r'$x_1$', zlabel=r'$u^*$', title=f'$u^*$(x(.),{depth})')

ax4 = fig.add_subplot(3,3,3, projection='3d')
ax4.plot_surface(x2, x1, err_x1, alpha=.5, label=r'$||x_1||$')
ax4.set(xlabel=r'$x_2$', ylabel=r'$x_1$', zlabel=r'$||x_1||$',
        title=f'$\max||x_1||$, k={kp}, {inval_x1:.2f}\% truncated')
ax4.plot_surface(x2, x1, np.ones((x2.shape[0], x1.shape[1]))*x1[1,1], alpha=.7)
ax4.plot_surface(x2, x1, np.ones((x2.shape[0], x1.shape[1]))*x1[-1,-1], alpha=.7)

ax5 = fig.add_subplot(3,3,6, projection='3d') # frameon=False
ax5.plot_surface(x2, x1, err_x2, alpha=.5, label=r'$||x_2||$')
ax5.set(xlabel=r'$x_2$', ylabel=r'$x_1$', zlabel=r'$||x_2||$',
        title=f'$||x_2||$, k={kp}, {inval_x2:.2f}\% truncated')
ax5.plot_surface(x2, x1, np.ones((x2.shape[0], x1.shape[1]))*x2[1,1], alpha=.7)
ax5.plot_surface(x2, x1, np.ones((x2.shape[0], x1.shape[1]))*x2[-1,-1], alpha=.7)

for i in range(ax.shape[0]):
    ax[i,0].grid(color='lightgray', linestyle='--')
    ax[i,0].legend()
    ax[i,1].axis('off')
    ax[i,2].axis('off')
fig.suptitle(f'Finite-time Optimal Cost (via DP): {Vz0}')
fig.show()
