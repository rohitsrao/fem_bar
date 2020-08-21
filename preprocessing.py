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

