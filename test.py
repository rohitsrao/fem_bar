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

#Creating a material
cm = Material(name='customMat', E=50000)

print('Material {} has been created successfully'.format(cm.name))
print('Young\'s Modulus: {}MPa'.format(cm.E))
print()

#Creating nodes from csv
f = './nodes.csv'
Node.create_nodes_from_csv(f)
Node.display_nodes()
