#This contains all classes for different kinds of solvers

#Libraries
import pandas as pd
import numpy as np

from preprocessing import *

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

        #Generating the reduced force vector for the truss
        self.truss.generate_reduced_force_vec()

        #Calling Truss.prep_for_solving
        #Initialises certain things
        self.truss.prep_for_solving()

        #List to store u1, u2, f2, k11
        data_dump = []

        #Defining a flag to keep track if no convergence was attained after 100 iterations
        _100IterationConvergedFlag = False

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
            print('Pseduo time: {:.4f}'.format(t))
            print('-------------------------')
            
            #Calculating the external force vector to be applied
            #for this increment
            ext_force = t*self.truss.Fr_total
            
            #Initialise residue vector for this incremenet
            res_vec = ext_force-int_force
            
            #Compute the norm of the residue vector
            res_norm = np.linalg.norm(res_vec)
            
            #Initialising the iteration counter
            i = 0
            
            #ITERATION LOOP
            while res_norm > cnvrg_delta:
            
                #Increment iteration counter
                i += 1
                
                #Loop through each element of the truss and 
                #compute the element stiffness matrix
                for e in self.truss.edict.values():
                    e.generate_stiffness_matrix()
                
                #Apply residue to nodes
                self.truss.apply_residue_to_nodes(res_vec)
                
                #Generate reduced stiffness matrix and reduced load vector
                self.truss.assemble_reduced_stiffness()
                self.truss.generate_reduced_force_vec()
                
                #Solve reduced system to get displacement for the current iteration
                self.truss.solve_elastic()
                
                #Update dofs after solving
                self.truss.update_dofs()
                
                #Looping through elements
                for e in self.truss.edict.values():

                    #Compute the degree of freedom vector
                    e.compute_dof_vec()
                    
                    #Transform global displacements into axial displacements
                    e.compute_axial_displacements()
                   
                    #Compute the strain in the element
                    e.compute_strain()
                    
                    #Compute stresses at gauss points
                    e.compute_stress()
                    
                    #Compute internal force in element
                    e.compute_internal_force()
                
                #Assemble the internal force vector for the truss
                self.truss.assemble_internal_force()
                
                #Set the newton raphson int_force variable
                int_force = self.truss.reduced_int_force
                
                #Update residue vector
                res_vec = ext_force - int_force
                
                #Compute norm of residue vector
                res_norm = np.linalg.norm(res_vec)
                
                #Print Iteration number and residue
                print('Iteration: {}    res_norm: {:.4E}'.format(i, res_norm))
                
                #Breaking from the loop if 100 iterations have passed and no convergence was attained
                if i == 100:
                    _100IterationConvergedFlag = True
                    print('Convergence not attained even after 100 iterations. Exiting Solver')
                    break
             
            #Extract the stiffness information
            #tmp_u1 = self.truss.edict[1].dof_vec[0, 0]
            #tmp_u2 = self.truss.edict[1].dof_vec[2, 0]
            #tmp_f = self.truss.edict[1].int_force[2, 0]
            #tmp_k11 = self.truss.edict[1].k_local_2d[0, 0]
            #tmp_delu = tmp_u2-tmp_u1
            #tmp_fbydelu = tmp_f/tmp_delu
            #row = [tmp_u1, tmp_u2, tmp_f, tmp_k11, tmp_delu, tmp_fbydelu]
            #data_dump.append(row)

            #Print blank line at end of increment
            print()

            #Checking if 100 iterations over
            if _100IterationConvergedFlag==True:
                break

        #Converting data_dump list to pandas dataframe
        #df = pd.DataFrame(data_dump, columns=['u1', 'u2', 'f', 'k11', 'delta_u', 'f_by_delta_u'])

        #Saving dataframe to csv
        #outfile = './data_dump_0.1N.csv'
        #df.to_csv(outfile, index=False)
