# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:56:12 2018

@author: LEAMERJM1
"""

from dmm import DMM
import numpy as np
from scipy import sparse
from numpy import linalg
from scipy.integrate import ode
from types import MethodType, FunctionType
import matplotlib.pyplot as plt

def rhs(beta, rho0, scaledH, rho, i, j):
    '''
    Input:
        beta - time step that the function is being called on
        rho0 - element of the density matrix that is being propagated
        scaledH - scaled Hamiltonian operator
        rho - the entire matrix
        i, j -  the indices identifying the appropriate matrix element
    Output:
        f - the next value of the matrix element
    '''
    identity = np.identity(rho.shape[0], dtype=complex)
    K = scaledH.dot(identity - rho)
    Kdagger = K.conj().T
    f = 0.0
    for k in range(rho.shape[0]):
        f += K[i,k]*rho[k,j] + rho[i,k]*Kdagger[k,j] 
        
    f += 2*rho[i,j]*(K.dot(rho)).trace().sum()
    return f
    
def rhsmatrix(beta, rho, scaledH):
    '''
    Input:
        beta - time step that the function is being called on
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
    
def main():
    np.random.seed(4936601)
    rows = int(input("Enter the number of rows in the matrix: "))
    columns = int(input("Enter the number of columns in the matrix: "))
    dmm = DMM(
        dbeta=1.,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(rows,columns)) + 1j * np.random.normal(size=(rows,columns)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )
    scaledH = dmm.H - dmm.mu * dmm.identity
    scaledH *= -0.5
    rhocopy = dmm.rho.copy()
    print(rhocopy.trace())
    
    #Test rhs and rhs matrix
    print("Testing rhs and rhs matrix:")
    rho0 = dmm.rho.copy()
    rho1 = dmm.rho.copy()
    beta = dmm.beta
    dbeta = dmm.dbeta
    numsteps = 3000
    #rows = dmm.rho.shape[0]
    #columns = dmm.rho.shape[1]
    
    '''
    #Test rhs and rhs matrix
    print("Testing rhs and rhs matrix:")
    
    #test rhs
    istep = 0
    while(istep < numsteps):
        for i in range(rows):
            for j in range(columns):
                rho0[i,j] += dbeta*rhs(beta, rho0[i,j], scaledH, rho0, i, j)
        istep += 1
    rhs_eigvals = linalg.eigvalsh(rho0)
    print("rhs_eigvals:")
    print(rhs_eigvals)
    
    #test rhsmatrix
    istep = 0
    while(istep < numsteps):
        rho1 += dbeta*rhsmatrix(beta, rho1, scaledH)
        istep += 1
    rhsmatrix_eigvals = linalg.eigvalsh(rho1)
    print("rhsmatrix_eigvals:")
    print(rhsmatrix_eigvals)
    '''
    rows = int(np.sqrt(rho0.size))
    #solver = np.array([[ode(rhs) for j in range(columns)] for i in range(rows)])
    solver = ode(rhsmatrix).set_integrator('zvode',method='bdf')
    solver.set_initial_value(rho0.reshape(-1), 0.).set_f_params(scaledH)
    while solver.successful() and solver.t < dmm.dbeta*numsteps:
        solver.integrate(solver.t + dmm.dbeta)
    
    rho0 = solver.y.reshape(rows,rows)
    print(rho0.trace())
    print("Norm: ", np.linalg.norm(rho0-rho0.conj().T))
    
    '''
    for i in range(rows):
        for j in range(columns):
            solver[i,j].set_integrator('zvode', method='bdf')
            solver[i,j].set_initial_value(rho0[i,j], beta0)
            solver[i,j].set_f_params(scaledH, rho0, i, j)
            k = 1
            while solver[i,j].successful() and solver[i,j].t < betamax:
                solver[i,j].integrate(betas[k])
                k += 1
            rhocopy[i,j] = solver[i,j].y[0]
            #finalbetas[i,j] = solver[i,j].t
    '''
    #print("Zvode Results:")
    #print(finalbetas)
    #print(rhocopy)
    
    dmm.rk4(dmm.deriv, numsteps)
    print("Trace = " + str(dmm.rhocopy.trace()))
    #print("RK4 method:")
    #print(dmm.rhocopy)
    
    dmm.beta = 0.0
    dmm.rk1(dmm.deriv, numsteps)
    #print("RK1 method:")
    #print(dmm.rhocopy)
    
    #rho_exact = linalg.eigvalsh(dmm.get_exact_DF())[::-1]
    #print("Rho_exact: " + str(rho_exact))
    exact_pop = dmm.get_exact_pop()
    print("Exact_pop: " + str(exact_pop))
    rho_rk4 = linalg.eigvalsh(dmm.rhocopy)[::-1]
    print("Rho_rk4: " + str(rho_rk4))
    rho_rk1 = linalg.eigvalsh(dmm.rhocopy)[::-1]
    print("Rho_rk1: " + str(rho_rk1))
    rho_zvode = linalg.eigvalsh(rho0)[::-1]
    print("Rho_zvode: " + str(rho_zvode))
    E = dmm.E
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Population at $\\beta = %.2f$, $\mu = %.2f$" % (dmm.beta, dmm.mu))
    ax1.set_xlabel("Energy")
    ax1.set_ylim([-0.2,1.2])
    ax1.set_ylabel("Population")
    ax1.plot(E, rho_rk4, '*-', label='RK4')
    #ax1.plot(E, rho_exact, '*-', label='EXPM')
    ax1.plot(E, rho_zvode, '*-', label='Zvode')
    ax1.plot(E, exact_pop, '*-', label='Pop')
    ax1.legend(numpoints=1)
    plt.show()
    
    
    
if __name__ == '__main__':
    #print(main.__doc__)
    main()
