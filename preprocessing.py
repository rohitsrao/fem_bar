#This contains all classes required for preprocessing.

#Class for Bar
class Bar():

    #Initializer
    def __init__(self, mat):

        #Setting the material of the bar
        self.mat = mat

#Class for Degree of Freedom
class DOF():

    #Creating a class variable to count number of DOF objects created
    #Also serves as DOF id
    count = 0

    def __init__(self):

        #Incrementing DOF object counter
        DOF.count += 1

        #Setting DOF id
        self.id = DOF.count

        #Initialising a degree of freedom in x-direction
        self.u = 0.0

#Class for material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the material
        #typecasts user value for E into float
        self.E = float(E)

#Class for a node
class Node():

    #Defining a counter as class variable
    #Keeps count of number of nodes created
    #Counter value serves as node id
    count = 0

    #Creating a dictionary to store created nodes
    #keys are the node ids
    ndict = {}

    #Initializer
    def __init__(self, x, y):

        #Incrementing node counter
        Node.count += 1

        #Setting node id
        self.id = Node.count

        #Defining the coordinates of the node
        #Typecasting them to float to be sure
        self.x = float(x)
        self.y = float(y)

        #Adding the created node to ndict
        Node.ndict[self.id] = self

    #Program to display all nodes
    @classmethod
    def display_nodes(cls):

        print('Node ID   X   Y')
        for n in cls.ndict:
            line = str(n) + str(cls.ndict[n].x) + str(cls.ndict[n].y)
            print(line)

    #Must format the output of the nodes in a proper string column

