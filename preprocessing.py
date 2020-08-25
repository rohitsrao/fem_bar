#This contains all classes required for preprocessing.

#Libraries
import pandas as pd

#Class for Bar Element
class Element():

    #Defining a counter as class variable
    #Keeps count of number of elements created
    #Counter serves as element id
    count = 0

    #Creating a dictionary to store created elements
    edict = {}    

    #Initializer
    def __init__(self, n1, n2):

        '''
        Create a bar element between two nodes n1 and n2

        Inputs - 
        n1 - Node - first node of the element
        n2 - Node - second node of the element
        '''

        #Incrementing counter
        Element.count += 1

        #Setting element id
        self.id = Element.count

        #Creating a dictionary to store nodes belonging to an element
        self.n = {}

        #Adding node objects to the dictionary
        self.n[0] = n1
        self.n[1] = n2

        #Adding the created element to the edict dictionary
        Element.edict[self.id] = self


#Class for material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the material
        #typecasts user value for E into float
        self.E = float(E)

#Class for a Node
class Node():

    #Defining a counter as class variable
    #Keeps count of number of nodes created
    #Counter value serves as node id
    count = 0

    #Creating a dictionary to store created nodes
    #keys are the node ids
    ndict = {}

    #Setting class variables needed for display_nodes method
    col_width = 7
    col_pad = 3

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

        #Initializing degrees of freedom for this node
        self.ux = None
        self.uy = None

        #Adding the created node to ndict
        Node.ndict[self.id] = self

    #Alternative initializer to create nodes from csv
    @classmethod
    def create_nodes_from_csv(cls, f):

        '''
        This method acts as an alternative initializer for the Node class
        and provides a way to create multiple node objects from 
        coordinate data of nodes from a csv

        Inputs:
        f - string - path to csv file

        Note:
        csv file should have columns x, y
        '''

        #Creating a dataframe from the imported data from csv
        df = pd.read_csv(f)

        #Compute the number of rows in dataframe
        num_rows = df.shape[0]

        #Loop through the rows
        for i in range(num_rows):

            #Extract the coordinate from a single row
            x = df.iloc[i]['x']
            y = df.iloc[i]['y']

            #Create node by calling initializer
            cls(x, y)

    #Program to display all nodes
    @classmethod
    def display_nodes(cls):

        #Computing total cell length - width + padding
        cell_width = Node.col_width + Node.col_pad

        #Column names
        col_names = ['Node ID', 'X', 'Y', 'ux', 'uy']
        print(''.join(name.ljust(cell_width) for name in col_names))

        #Horizontal line below column name
        hl = '-'*Node.col_width
        print(''.join(hl.ljust(cell_width) for name in col_names))

        #Looping through each created node
        for n in cls.ndict:

            #Creating a row of the output table
            row = [str(n)]
            row.append(str(cls.ndict[n].x))
            row.append(str(cls.ndict[n].y))
            row.append(str(cls.ndict[n].ux))
            row.append(str(cls.ndict[n].uy))
            print(''.join(cell.ljust(cell_width) for cell in row))
        
        #Final print to create space
        print()

