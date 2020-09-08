#This is a simple python script to test code as it is being developed

#Defining complete FEM workflow
#Define Material
#Define Nodes from CSV
#Define which nodes go into which element from csv
#Create the bar elements
#Assemble them into a truss
#Defing the boundary conditions on the truss
#Define loads on the truss
#Generate element stiffness matrix
#Assemble global stiffness matrix
#Apply boundary condition
#Solve

import pandas as pd

from preprocessing import *
from solvers import NewtonRaphson

#Intialising
Element.stiffness_integrand_generator()
Element.set_transformation_matrix_sym()

#Creating a material
mat1 = Material(name='mat1', E=50000)

print('Material {} has been created successfully'.format(mat1.name))
print('Young\'s Modulus: {}MPa'.format(mat1.E))
print()

#Creating nodes from csv
f = './nodes.csv'
Node.create_nodes_from_csv(f)
Node.display_nodes()


#Creating elements from csv
f = './elements.csv'
Element.create_elements_from_csv(f)
Element.set_material(mat1)
Element.display_elements()

truss = Truss()
truss.assemble_full_stiffness()

#Apply loads from csv
f = './loads.csv'
truss.apply_loads_from_csv(f)
Node.display_nodes()

#Apply bcs from csv
f = './bcs.csv'
truss.apply_bcs_from_csv(f)
Node.display_nodes()

truss.assemble_reduced_stiffness()
truss.generate_reduced_force_vec()
truss.prep_for_solving()
truss.solve_elastic()
truss.update_dofs()
Node.display_nodes()

