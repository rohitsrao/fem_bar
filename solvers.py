#This contains all classes for different kinds of solvers

class NewtonRaphson():

    def __init__(self, truss):

        #Storing a reference to the Truss object being solved
        self.truss = truss

    def solve(self, num_increments, cnvrg_delta=1e-8):
        '''
        Function that implements the NewtonRaphson Scheme

        Inputs
        num_increments - int - number of increments
        cnvrg_delta - float - acceptable error in residue for convergence
        '''

        #Set pseduo time to zero
        t = 0

        #Compute number of time increments
        delta_t = 1/num_increments

        #Initialising a counter variable to keep track of the
        #number of increments
        n = 0

        #Computing shape of reduced internal force vector
        num_nodes_total = len(self.truss.ndict)
        num_nodes_bc = len(self.truss.bc_dof_ids)

        print('t: {}'.format(t))
        print('delta_t: {}'.format(delta_t))
        print('n: {}'.format(n))
        print('num_nodes_total: {}'.format(num_nodes_total))
        print('num_nodes_bc: {}'.format(num_nodes_bc))
        print()


