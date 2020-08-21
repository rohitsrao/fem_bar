#This contains all classes required for preprocessing.

#Creating a class for Bar
class Bar():

    #Initializer
    def __init__(self, mat):

        #Setting the material of the bar
        self.mat = mat

#Creating a class for material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the material
        #typecasts user value for E into float
        self.E = float(E)

#Creating a class for a node
class Node():

    #Initializer
    def __init__(self, x, y):

        #Defining the coordinates of the node
        #Typecasting them to float to be sure
        self.x = float(x)
        self.y = float(y)
