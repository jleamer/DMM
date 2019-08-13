# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:58:18 2019

@author: LEAMERJM1
"""

import time
from dmm import DMM
import numpy as np
from scipy import sparse
from numpy import linalg
from scipy.integrate import ode
from types import MethodType, FunctionType
import matplotlib.pyplot as plt

def rhsmatrix(beta, rho, scaledH):
    '''
    This function implements the rhs of the derivative expression 
    for use in the python ODE Solvers
    Input:
        beta - time step that the function is being called on; not actually used
        rho - the matrix that is being propagated
        scaledH - scaled Hamiltonian operator
    Output:
        f - the derivative of the matrix
    '''
    rows = int(np.sqrt(rho.size))
    rho = rho.reshape(rows,rows)
    identity = np.identity(rho.shape[0], dtype=complex)
    K = scaledH.dot(identity - rho)
    f = K.dot(rho) + rho.dot(K.conj().T) - 0*rho*(K.dot(rho)).trace().sum()
    return f.reshape(-1)

        
if __name__ == '__main__':
    ###########################################################################
    #
    #   Generate some examples of dmm to time
    #
    ###########################################################################
    
    numsteps = 3000
    runs = 1
    numobjects = 10
    dmm_list = np.empty(numobjects, dtype=object)
    multiple = 100
    
    for i in range(numobjects):
        rows = multiple*(i+1)
        cols = multiple*(i+1)
        
        dmm_list[i] = DMM(
                dbeta=0.003,
                dmu=0.0005,
                mu=-0.9,
                # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
                #H=sparse.random(70, 70, density=0.1),
                H=np.random.normal(size=(rows, cols)) + 1j * np.random.normal(size=(rows, cols)),
                #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
                )
    ###########################################################################
    #
    #   Time Zvode solving
    #
    ###########################################################################
    
    rho_zvode_list = np.empty(numobjects, dtype=object)
    solvers = np.empty(numobjects, dtype=object)
    scaledH_list = np.empty(numobjects, dtype=object)
    zvode_eigvals = np.empty(numobjects, dtype=object)
    time_list = np.array([[0.0 for i in range(runs)] for i in range(numobjects)])
    average_times = np.array([0.0 for i in range(numobjects)])    
    std_dev_times = np.array([0.0 for i in range(numobjects)])
    
    for i in range(numobjects):
        rho_zvode_list[i] = dmm_list[i].rho.copy()
        scaledH_list[i] = -0.5*(dmm_list[i].H - dmm_list[i].mu * dmm_list[i].identity)
        print("Object #: ", i)
        
        for j in range(runs):
            solvers[i] = ode(rhsmatrix).set_integrator('zvode', method='bdf')
            solvers[i].set_initial_value(rho_zvode_list[i].reshape(-1), 0.).set_f_params(scaledH_list[i])
        
            time_i = time.clock()
            #print("i = ", time_i)
            while solvers[i].successful() and solvers[i].t < dmm_list[i].dbeta*numsteps:
                solvers[i].integrate(solvers[i].t + dmm_list[i].dbeta)
            time_f = time.clock()
            #print("f = ", time_f)
                
            time_list[i][j] = time_f - time_i
        
        #print(time_list[i])
        average_times[i] = np.mean(time_list[i])
        print("Average[", i, "]: %.6e" % average_times[i])
    
        std_dev_times[i] = np.std(time_list[i])
        print("Standard Dev.[", i, "]: %.6e" % std_dev_times[i])
    
    sizes = np.array([multiple*(i+1) for i in range(numobjects)])
    title = "Time for Zvode (" + str(sizes[0]) + "-" + str(sizes[numobjects-1]) + ")"
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    ax1.set_title(title)
    ax1.set_xlabel('Size (Rows)')
    ax1.set_ylabel('Time (s)')
    ax1.plot(sizes, average_times, 'o-', label='Average Time')
    ax1.plot(sizes, std_dev_times, 'o-', label='Std. Dev')
    ax1.legend(numpoints = 1)