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

    #Defining a counter as class variable
    #Keeps count of number of nodes created
    #Counter value serves as node id
    count = 0

    #Initializer
    def __init__(self, x, y):

        #Incrementing node counter
        Node.count += 1

        #Setting node id
        self.id = count

        #Defining the coordinates of the node
        #Typecasting them to float to be sure
        self.x = float(x)
        self.y = float(y)
