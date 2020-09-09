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
Element.symbolic_quantities_generator()
Element.generate_transformation_matrix_sym()

#Creating a material
sig_poly = np.poly1d([47940.7970, -20382.2400, 3018.3718, 94.0444])
yp = 0.002
mat1 = Material(name='mat1', E=50000, sig_poly=sig_poly, yp=yp)

print('Material {} has been created successfully'.format(mat1.name))
print()

Material.display_material_data()

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

#truss.assemble_reduced_stiffness()
#truss.generate_reduced_force_vec()
#truss.prep_for_solving()
#truss.solve_elastic()
#truss.update_dofs()
#Node.display_nodes()

#Creating solver object
solver = NewtonRaphson(truss)
solver.solve(num_increments=3)
