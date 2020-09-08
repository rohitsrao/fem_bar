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

        #INCREMENT LOOP
        #Applying loads until t=1
        #this is checked by subtracting current value of t from 1 and seeing if it is
        #greater than 1e-12. This needs to be done as delta_t can be fractions
        while (1-t)>1e-12:

            #Increment the increment counter
            n += 1

            #Increment time
            t += delta_t

            #Display current increment
            print('Increment: {}'.format(n))
            print('Pseduo time: {}'.format(t))
            print('-------------------------')

            #Calculating the external force vector to be applied
            #for this increment
            ext_force = t*self.truss.Fr

            #Initialise residue vector for this incremenet
            res_vec = ext_force-int_force

            #Compute the norm of the residue vector
            res_norm = np.linalg.norm(res_vec)

            #Initialising the iteration counter
            i = 0

            #ITERATION LOOP
            while res_norm > cnvrg_delta:
                print("boo yeah")
                break
