#This contains all classes required for preprocessing.

#Libraries
import numpy as np
import pandas as pd
import sympy as sp

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

        #Creating a placeholder for material
        self.mat = None

        #Creating a dictionary to store nodes belonging to an element
        self.n = {}

        #Adding node objects to the dictionary
        self.n[0] = n1
        self.n[1] = n2

        #Adding the created element to the edict dictionary
        Element.edict[self.id] = self

        #Computing element geometry
        self.compute_geometry()

    @classmethod
    def create_elements_from_csv(cls, f):

        '''
        This method acts as an alternative initializer for the Element class
        and provides a way to create multiple Element objects from nodal list
        in a csv

        Inputs:
        f - string - path to csv file

        Note:
        csv file should have columns n1, n2 for node ids
        '''

        #Creating a dataframe from the imported data from csv
        df = pd.read_csv(f)

        #Compute the number of rows in dataframe
        num_rows = df.shape[0]

        #Loop through the rows
        for i in range(num_rows):

            #Extract the node ids from a single row
            n1_id = df.iloc[i]['n1']
            n2_id = df.iloc[i]['n2']

            #Extracting the node objects corresponding to node ids
            n1 = Node.ndict[n1_id]
            n2 = Node.ndict[n2_id]

            #Create node by calling initializer
            cls(n1, n2)

        #Deleting dataframe after elements have been created
        del df

    @classmethod
    def define_symbolic_variables(cls):

        #Natural Coordinate
        cls.xi = sp.symbols('xi')

        #Material and geometry
        cls.E = sp.symbols('E')
        cls.L = sp.symbols('L')
        cls.A = sp.symbols('A')

        #Degrees of freedom of bar element
        cls.u1 = sp.symbols('u1')
        cls.u2 = sp.symbols('u2')

    @classmethod
    def display_elements(cls):

        #Defining column widths and padding
        col_width = 7
        col_pad = 3

        #Computing total cell length - width + padding
        cell_width = col_width + col_pad

        #Printing note
        print('Angles are measured counter clockwise at first node in an element')
        print()

        #Column names
        col_names = ['Elem ID', 'mat', 'n1', 'n2', 'len', 'deg']
        print(''.join(name.ljust(cell_width) for name in col_names))

        #Horizontal line below column name
        hl = '-'*col_width
        print(''.join(hl.ljust(cell_width) for name in col_names))

        #Looping through each created element
        for e in cls.edict.values():

            #Creating a row of the output table
            row = []
            row.append(str(e.id))
            row.append(e.mat.name)
            row.append(str(e.n[0].id))
            row.append(str(e.n[1].id))
            row.append(str(e.l))
            row.append(str(e.theta_deg))
            print(''.join(cell.ljust(cell_width) for cell in row))
        
        #Final print to create space
        print()

    @classmethod
    def set_material(cls, mat):
        '''
        Sets the material for every single element created

        Inputs - Material - material object
        '''

        #Looping through all the elements and setting material
        for e in cls.edict.values():
            e.mat = mat

    def compute_geometry(self):
        '''
        Method to compute the length and angle of the element
        '''

        #Extracting the coordinates
        #Node 1
        x1 = self.n[0].x
        y1 = self.n[0].y
        #Node 2
        x2 = self.n[1].x
        y2 = self.n[1].y

        #Computing length
        self.l = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        #Computing the angle
        m = (y2-y1)/(x2-x1)
        self.theta = np.arctan(m)

        #Converting angle to degrees for display
        self.theta_deg = np.degrees(self.theta)

#Class for material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the mateelement
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

        #Deleting dataframe after nodes have been created
        del df

    #Program to display all nodes
    @classmethod
    def display_nodes(cls):

        #Setting column width and padding
        col_width = 7
        col_pad = 3

        #Computing total cell length - width + padding
        cell_width = col_width + col_pad

        #Column names
        col_names = ['Node ID', 'X', 'Y', 'ux', 'uy']
        print(''.join(name.ljust(cell_width) for name in col_names))

        #Horizontal line below column name
        hl = '-'*col_width
        print(''.join(hl.ljust(cell_width) for name in col_names))

        #Looping through each created node
        for n in cls.ndict.values():

            #Creating a row of the output table
            row = []
            row.append(str(n.id))
            row.append(str(n.x))
            row.append(str(n.y))
            row.append(str(n.ux))
            row.append(str(n.uy))
            print(''.join(cell.ljust(cell_width) for cell in row))
        
        #Final print to create space
        print()

