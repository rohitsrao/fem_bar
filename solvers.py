#This contains all classes for different kinds of solvers

#Libraries
import numpy as np

class NewtonRaphson():

    def __init__(self, truss):

        #Storing a reference to the Truss object being solved
        self.truss = truss

    def solve(self, num_increments=1, cnvrg_delta=1e-8):
        '''
        Function that implements the NewtonRaphson Scheme

        Inputs
        num_increments - int - number of increments - default value: 1
        cnvrg_delta - float - acceptable error in residue for convergence - default value: 1e-8
        '''

        #Set pseduo time to zero
        t = 0

        #Compute number of time increments
        delta_t = 1/num_increments

        #Initialising a counter variable to keep track of the
        #number of increments
        n = 0

        #Initializing a reduced internal force vector
        int_force_shape = (self.truss.reduced_dimension, 1)
        int_force = np.zeros(shape=int_force_shape)

        print('t: {}'.format(t))
        print('delta_t: {}'.format(delta_t))
        print('n: {}'.format(n))
        print('int_force')
        print(int_force)
        print()


