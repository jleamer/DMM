# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 20:15:26 2019

@author: LEAMERJM1

DMM_stats.py - file for comparing our different dmm methods
"""

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
    #   Generate some examples to compare
    #
    ###########################################################################
    
    np.random.seed(4936601)
    numsteps = 3000
    numobjects = 3
    '''    
    dmm1 = DMM(
        dbeta=0.003,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )
    
    dmm2 = DMM(
        dbeta=0.003,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )
    '''
    dmm_list = np.empty(numobjects, dtype=object)
    for i in range(numobjects):
        dmm_list[i] = DMM(
                dbeta=0.003,
                dmu=0.0005,
                mu=-0.9,
                # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
                #H=sparse.random(70, 70, density=0.1),
                H=np.random.normal(size=(40,40)) + 1j * np.random.normal(size=(40,40)),
                #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
                )
        
    
    ###########################################################################
    #
    #   Compare propagate methods
    #
    ###########################################################################         
    '''
    dmm1.propagate_beta1(numsteps)
    exact_dmm1 = dmm1.get_exact_pop()
    prop_dmm1 = linalg.eigvalsh(dmm1.rhocopy)[::-1]
    
    dmm2.propagate_beta1(numsteps)
    exact_dmm2 = dmm2.get_exact_pop()
    prop_dmm2 = linalg.eigvalsh(dmm2.rhocopy)[::-1]
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax11 = fig1.add_subplot(121)
    ax11.set_title("Population of DMM1 at $\\beta = %.2f$, $\mu = %.2f$" % (dmm1.beta, dmm1.mu))
    ax11.set_xlabel('Population')
    ax11.set_ylabel('Energy')
    ax11.set_ylim([-0.2, 1.2])
    ax11.plot(dmm1.E, exact_dmm1, '*-', label='DMM1 exact')
    ax11.plot(dmm1.E, prop_dmm1, '*-', label='DMM1 prop')
    ax11.legend(numpoints = 1)
    
    ax12 = fig1.add_subplot(122)
    ax12.set_title("Population of DMM2 at $\\beta = %.2f$, $\mu = %.2f$" % (dmm2.beta, dmm2.mu))
    ax12.set_xlabel('Population')
    ax12.set_ylabel('Energy')
    ax12.set_ylim([-0.2, 1.2])
    ax12.plot(dmm2.E, exact_dmm2, '*-', label='DMM2 exact')
    ax12.plot(dmm2.E, prop_dmm2, '*-', label='DMM2 prop')
    ax12.legend(numpoints = 1)
    '''
    exact_dmm = np.empty(numobjects, dtype=object)
    prop_dmm = np.empty(numobjects, dtype=object)
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Population')
    ax1.set_ylabel('Energy')
    ax1.set_ylim([-0.2, 1.2])
    
    for i in range(numobjects):
        dmm_list[i].propagate_beta1(numsteps)
        prop_dmm[i] = linalg.eigvalsh(dmm_list[i].rhocopy)[::-1]
        exact_dmm[i] = dmm_list[i].get_exact_pop()
        ax1.plot(dmm_list[i].E, exact_dmm[i], '*-', label='Exact')
        ax1.plot(dmm_list[i].E, prop_dmm[i], '*-', label='Prop')
    ax1.set_title("Population of DMM at $\\beta = %.2f$, $\mu = %.2f$" % (dmm_list[0].beta, dmm_list[0].mu))
    ax1.legend(numpoints = 1)
    
    
    ###########################################################################
    #
    #   Compare RK4 methods
    #
    ###########################################################################
    '''
    dmm1.beta = 0.0
    dmm1.rk4(dmm1.deriv, numsteps)
    dmm1_rk4_eigvals = linalg.eigvalsh(dmm1.rhocopy)[::-1]
    
    dmm2.beta = 0.0
    dmm2.rk4(dmm2.deriv, numsteps)
    dmm2_rk4_eigvals = linalg.eigvalsh(dmm2.rhocopy)[::-1]
    
    
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("RK4 Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm1.beta, dmm1.mu))
    ax2.set_xlabel('Population')
    ax2.set_ylabel('Energy')
    ax2.set_ylim([-.2, 1.2])
    ax2.plot(dmm1.E, dmm1_rk4_eigvals, '*-', label='DMM 1')
    ax2.plot(dmm2.E, dmm2_rk4_eigvals, '*-', label='DMM 2')
    ax2.legend(numpoints = 1)
    '''
    rk4_eigvals = np.empty(numobjects, dtype=object)
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Population')
    ax2.set_ylabel('Energy')
    ax2.set_ylim([-0.2, 1.2])
    
    for i in range(numobjects):
        dmm_list[i].beta = 0.0 #resets the beta value for the RK4 method
        dmm_list[i].rk4(dmm_list[i].deriv, numsteps)
        rk4_eigvals[i] = linalg.eigvalsh(dmm_list[i].rhocopy)[::-1]
        ax2.plot(dmm_list[i].E, rk4_eigvals[i], '*-', label='RK4')
        ax2.plot(dmm_list[i].E, exact_dmm[i], '*-', label='Exact')
    ax2.set_title("RK4 Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm_list[0].beta, dmm_list[0].mu))
    
    
    ###########################################################################
    #
    #   Compare ODE Solver methods
    #
    ###########################################################################
    '''
    rho_zvode1 = dmm1.rho.copy()
    scaledH1 = dmm1.H - dmm1.mu * dmm1.identity
    scaledH1 *= -0.5 
    solver1 = ode(rhsmatrix).set_integrator('zvode', method='bdf')
    solver1.set_initial_value(rho_zvode1.reshape(-1), 0.).set_f_params(scaledH1)
    while solver1.successful() and solver1.t < dmm1.dbeta*numsteps:
        solver1.integrate(solver1.t + dmm1.dbeta)
    dmm1_zvode_eigvals = linalg.eigvalsh(solver1.y.reshape(40,40))[::-1]    
        
    rho_zvode2 = dmm2.rho.copy()
    scaledH2 = dmm2.H - dmm2.mu * dmm2.identity
    scaledH2 *= -0.5 
    solver2 = ode(rhsmatrix).set_integrator('zvode', method='bdf')
    solver2.set_initial_value(rho_zvode2.reshape(-1), 0.).set_f_params(scaledH2)
    while solver2.successful() and solver2.t < dmm2.dbeta*numsteps:
        solver2.integrate(solver2.t + dmm2.dbeta)
    dmm2_zvode_eigvals = linalg.eigvalsh(solver2.y.reshape(40,40))[::-1]
    '''
    rho_zvode_list = np.empty(numobjects, dtype=object)
    solvers = np.empty(numobjects, dtype=object)
    scaledH_list = np.empty(numobjects, dtype=object)
    zvode_eigvals = np.empty(numobjects, dtype=object)
    identity = dmm_list[0].identity
    
    fig3 = plt.figure(3)
    fig3.clf()
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel('Population')
    ax3.set_ylabel('Energy')
    ax3.set_ylim([-.2, 1.2])
    
    for i in range(numobjects):
        rho_zvode_list[i] = dmm_list[i].rho.copy()
        scaledH_list[i] = -0.5*(dmm_list[i].H - dmm_list[i].mu * identity)
        
        solvers[i] = ode(rhsmatrix).set_integrator('zvode', method='bdf')
        solvers[i].set_initial_value(rho_zvode_list[i].reshape(-1), 0.).set_f_params(scaledH_list[i])
        while solvers[i].successful() and solvers[i].t < dmm_list[i].dbeta*numsteps:
            solvers[i].integrate(solvers[i].t + dmm_list[i].dbeta)
        
        zvode_eigvals[i] = linalg.eigvalsh(solvers[i].y.reshape(40,40))[::-1]
        
        ax3.plot(dmm_list[i].E, zvode_eigvals[i], '*-', label='Zvode')
        ax3.plot(dmm_list[i].E, exact_dmm[i], '*-', label='Exact')
        
    ax3.legend(numpoints = 1)
    ax3.set_title("Zvode Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm_list[0].beta, dmm_list[0].mu))
    
    ###########################################################################
    #
    #   Compare Zvode method to Propagate method
    #
    ###########################################################################
    '''
    print("Error norm b/w zvode and exact diagonalization for dmm1: %.2e" % np.linalg.norm(dmm1_zvode_eigvals - exact_dmm1))
    print("Error norm b/w propagation and exact diagonalization for dmm1: %.2e" % np.linalg.norm(prop_dmm1 - exact_dmm1))
    
    print("Error norm b/w zvode and exact diagonalization for dmm2: %.2e" % np.linalg.norm(dmm2_zvode_eigvals - exact_dmm2))
    print("Error norm b/w zvode and exact diagonalization for dmm2: %.2e" % np.linalg.norm(prop_dmm2 - exact_dmm2))
    '''
    for i in range(numobjects):
        zvode_err = np.linalg.norm(zvode_eigvals[i] - exact_dmm[i])
        prop_err = np.linalg.norm(prop_dmm[i] - exact_dmm[i])
        print("Error norm b/w zvode and exact for dmm[", i, "]: %.2e" % zvode_err)
        print("Error norm b/w prop and exact for dmm[", i, "]: %.2e" % prop_err)
        print("Difference in errors: %.2e" % (zvode_err - prop_err))
        print("")