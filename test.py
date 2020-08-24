#This is a simple python script to test code as it is being developed

from preprocessing import *

#Creating a material
cm = Material(name='customMat', E=50000)

print('Material {} has been created successfully'.format(cm.name))
print('Young\'s Modulus: {}MPa'.format(cm.E))
print()

#Creating nodes
n1 = Node(0, 0)

#Checking
Node.display_nodes()
