# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:56:12 2018

@author: LEAMERJM1

Notes:
    Increasing the step size to 0.03 caused Zvode to match perfectly with the exact pop
    and made RK4 match more closely than at a step size of 0.003 both when the TP element
    was turned off.
    
    Increasing the number of steps to 30000 caused Zvode to match perfectly and made
    RK4 have a more pronounced slope both when the TP element was turned off.
    
    It appears that each step of RK4 causes a very small change to the eigenvalues of rho.
    Maybe that's why you need to increase the step size to get a better result?  It
    causes the eigenvalues to change more quickly?
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
    
def jacobian(beta, rho, scaledH):
    '''
    Input:
        beta - time step the function is being called on; not actually used
        rho - the matrix that is being propagated
        scaledH - scaled Hamiltonian operator
    Output:
        f - the jacobian of rho
        
    Not sure if we ever finished implementing this.  I don't think we did
    '''
    rows = int(np.sqrt(rho.size))
    identity = np.identity(rho.shape[0], dtype=complex)
    K = scaledH.dot(identity-rho)
    jacobian = np.zeros((2*rows, 2))
    #print(jacobian.shape[0])
    #print(jacobian.shape[1])
    #for i in range(jacobian.shape[0]):
     #   for j in range(jacobian.shape[1]):
      #      if(i%2==1):
       #         jacobian[i,j] = (K.dot(rho) + rho.dot(K.conj().T) - 0*rho*(K.dot(rho)).trace().sum())[i,j]            
    #print(jacobian)      
    
def main():
    np.random.seed(4936601)
    rows = int(input("Enter the number of rows in the matrix: "))
    columns = int(input("Enter the number of columns in the matrix: "))
    dmm = DMM(
        dbeta=0.003,
        dmu=0.0005,
        mu=-0.9,
        # randomly initialize Hamiltonian (symmetrization will take place in the constructor)
        #H=sparse.random(70, 70, density=0.1),
        H=np.random.normal(size=(rows,columns)) + 1j * np.random.normal(size=(rows,columns)),
        #np.random.rand(40, 40) + 1j * np.random.rand(40, 40)
    )
    scaledH = dmm.H - dmm.mu * dmm.identity
    scaledH *= -0.5
    rhocopy = dmm.rho.copy()#/dmm.rho.copy().trace()
    print("Trace: " + str(rhocopy.trace()))
    numsteps = 3000
    
    #jacobian(dmm.beta, rhocopy, scaledH) - not sure that we ever finished implementing this
    rows = int(np.sqrt(rhocopy.size))
    #solver = np.array([[ode(rhs) for j in range(columns)] for i in range(rows)])
    solver = ode(rhsmatrix).set_integrator('zvode', method='bdf')
    solver.set_initial_value(rhocopy.reshape(-1), 0.).set_f_params(scaledH)
    while solver.successful() and solver.t < dmm.dbeta*numsteps:
        solver.integrate(solver.t + dmm.dbeta)
    
    rho_ode = solver.y.reshape(rows,rows) #output of ode method above
    #print(rho_ode)
    print("Trace of zvode = " + str(rho_ode.trace())) #trace of rho after ode
    print("Norm: ", np.linalg.norm(rhocopy-rhocopy.conj().T)) #check if rho is Hermitian
    
    dmm.rk4(dmm.deriv, numsteps)
    print("Trace of rk4 = " + str(dmm.rhocopy.trace())) #trace of rho after rk4
    rho_rk4 = linalg.eigvalsh(dmm.rhocopy)[::-1] #eigenvalues of rho after rk4
    print("Rho_rk4: " + str(rho_rk4))
    #print("RK4 method:")
    #print(dmm.rhocopy)
    
    #rho_exact = linalg.eigvalsh(dmm.get_exact_DF())[::-1]
    #print("Rho_exact: " + str(rho_exact))
    
    exact_pop = dmm.get_exact_pop()
    print("Exact_pop: " + str(exact_pop))
    rho_zvode = linalg.eigvalsh(rho_ode)[::-1]
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
