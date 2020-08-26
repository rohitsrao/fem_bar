#This contains all classes required for preprocessing.

#Libraries
import numpy as np
import pandas as pd
import sympy as sp

from sympy.solvers.solveset import linsolve

#Class for DOF
class DOF():

    #Defining a counter as class variable
    #Keeps count of the number of elements created
    #Counter serves as dof id
    count = 0

    def __init__(self, symbol, value=None):
        '''
        Initializer for the DOF class

        Inputs 
        symbol - String - ux or uy
        value - Float value - By default set to None
        '''

        #Increment counter
        DOF.count += 1

        #Setting the symbol
        self.symbol = symbol

        #Setting the value
        self.value = value

#Class for Bar Element
class Element():

    #Defining a counter as class variable
    #Keeps count of number of elements created
    #Counter serves as element id
    count = 0

    #Creating a dictionary to store created elements
    edict = {}    

    #Initializer
    def __init__(self, n1, n2, A):

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

        #Setting the area of the element
        self.A = A

        #Creating a dictionary to store nodes belonging to an element
        self.n = {}

        #Adding node objects to the dictionary
        self.n[0] = n1
        self.n[1] = n2

        #Adding the created element to the edict dictionary
        Element.edict[self.id] = self

        #Computing element geometry
        self.compute_geometry()

        #Defining weights and points for gauss integration
        self.gp = [0]
        self.w = [2]

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
            
            #Extracing the area from the row
            A = df.iloc[i]['A']

            #Extracting the node objects corresponding to node ids
            n1 = Node.ndict[n1_id]
            n2 = Node.ndict[n2_id]

            #Create node by calling initializer
            cls(n1, n2, A)

        #Deleting dataframe after elements have been created
        del df

    @classmethod
    def stiffness_integrand_generator(cls):

        #Natural Coordinate
        cls.xi = sp.symbols('xi')

        #Material and geometry
        cls.E = sp.symbols('E')
        cls.L = sp.symbols('L')
        cls.A = sp.symbols('A')

        #Degrees of freedom of bar element
        cls.u1 = sp.symbols('u1')
        cls.u2 = sp.symbols('u2')

        #Defining Jacobian Matrix
        cls.J = sp.Matrix([[0.5*cls.L]])

        #Field Variable
        u = sp.symbols('u')

        #Defining coefficients for linear interpolation
        alpha0, alpha1 = sp.symbols(['alpha_0', 'alpha_1'])

        #Defining linear interpolation function
        u = alpha0 + alpha1*cls.xi

        #Substituting the values for xi at the nodes
        #Node1
        u1e = sp.Eq(cls.u1, u.subs(cls.xi, -1))
        #Node2
        u2e = sp.Eq(cls.u2, u.subs(cls.xi, 1))

        #Solving for the coefficients in terms of the nodal values of the
        #degrees of freedom using sp.solvers.solveset.linsolve. Returns a 
        #finite set. 
        res = linsolve([u1e, u2e], alpha0, alpha1)

        #Extracting the values of the Sympy FiniteSet into the variables
        #The elements the FiniteSet can be accessed via .args(index)
        #which gives a tuple containing sympy tuple
        alpha0_s = res.args[0][0]
        alpha1_s = res.args[0][1]

        #Substitute the coefficients back into the equation for the field variable
        u = u.subs([(alpha0, alpha0_s), (alpha1, alpha1_s)])

        #Group the terms of the nodal degrees of freedom
        u = sp.collect(sp.expand(u), [cls.u1, cls.u2])

        #Collecting the shape functions into separate variables
        N1 = u.coeff(cls.u1)
        N2 = u.coeff(cls.u2)

        #Defining the shape function matrix
        cls.N = sp.Matrix([[N1, N2]])

        #Defining the onezero matrix
        onezero = sp.Matrix([[1]])

        #Defining Gamma Matrix - inverse Jacobian
        gamma = cls.J.inv()

        #Defining DN Matrix
        DN = sp.diff(cls.N, cls.xi)

        #Computing the transformed strain displacement matrix Bt
        cls.Bt = onezero*gamma*DN

        #Defining the integrand
        cls.integrand = cls.E*cls.A*(cls.Bt.T*cls.Bt)*cls.J.det()

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
        col_names = ['Elem ID', 'mat', 'n1', 'n2', 'area', 'len', 'deg']
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
            row.append(str(e.A))
            row.append(str(e.L))
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

    @classmethod
    def set_transformation_matrix_sym(cls):
        '''
        This method is to set the symbolic transformation matrix
        to relate quantities from the local coordinate system to the
        global coordinate system
        '''
        
        #Defining a symbolic angle
        cls.phi = sp.symbols('phi')

        #Defining symbolic transformation matrix
        #T transforms quantities from local coordinate to global coordiante
        cls.T = sp.Matrix(np.zeros(shape=(4, 4)))
        cls.T[0, 0] = sp.cos(cls.phi)
        cls.T[0, 1] = -sp.sin(cls.phi)
        cls.T[1, 0] = sp.sin(cls.phi)
        cls.T[1, 1] = sp.cos(cls.phi)
        cls.T[2, 2] = sp.cos(cls.phi)
        cls.T[2, 3] = -sp.sin(cls.phi)
        cls.T[3, 2] = sp.sin(cls.phi)
        cls.T[3, 3] = sp.cos(cls.phi)

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
        self.L = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        #Computing the angle
        m = (y2-y1)/(x2-x1)
        self.theta = np.arctan(m)

        #Converting angle to degrees for display
        self.theta_deg = np.degrees(self.theta)

    def gauss_integrator(self, integrand):
        '''
        Performs gauss integration on integrand
        integrand must be a symbolic expression in xi (symbolic)
        '''

        #Defining a list to store the gauss integrals
        gauss_integrals = []

        #Looping through number of gauss points
        for i in range(len(self.gp)):

            sub_list = [(Element.xi, self.gp[i])]
            gauss_integral = self.w[i]*integrand.subs(sub_list)
            gauss_integrals.append(gauss_integral)

        #Calculating total integrand
        #Need to provide a start argument for the sum as symbolic matrix of zeros
        integral = sum(gauss_integrals, sp.Matrix(np.zeros(shape=integrand.shape)))

        return integral


    def generate_stiffness_matrix(self):
        '''
        This method is to generate the element stiffness matrix for the 
        1D bar via gauss integration
        '''

        #Compute the symbolic 1d stiffness matrix
        self.k_local_1d_s = self.gauss_integrator(Element.integrand)

        #Convert from symbolic matrix into numpy matrix
        sub_list = [(Element.E, self.mat.E), 
                    (Element.L, self.L),
                    (Element.A, self.A)]
        self.k_local_1d = self.k_local_1d_s.subs(sub_list)
        self.k_local_1d = np.array(self.k_local_1d).astype(np.float64)

        #Converting this 1d stiffness matrix into a 2D with zeros
        #for the appropriate terms in the y-direction
        #This is still in local coordinate system
        self.k_local_2d = np.zeros(shape=(4, 4))
        self.k_local_2d[0, 0] = self.k_local_1d[0, 0]
        self.k_local_2d[0, 2] = self.k_local_1d[0, 1]
        self.k_local_2d[2, 0] = self.k_local_1d[1, 0]
        self.k_local_2d[2, 2] = self.k_local_1d[1, 1]

    def transform_k_local_global(self):
        '''
        This method transforms the 2D stiffness matrix from the local coordinate
        system to the global coordinate system
        '''

        #Computing the transformation matrix for element
        #by substituting angle
        self.T = Element.T.subs(Element.phi, self.theta)
        self.T = np.array(self.T).astype(np.float64)

        #Transforming the 2d local stiffness matrix to 2d global stiffness matrix
        temp = np.matmul(self.k_local_2d, np.linalg.inv(self.T))
        self.k_global_2d = np.matmul(self.T, temp)

#Class for Load
class Load():

   #Defining a counter as class variable
   #Keeps count of number of loads created
   #counter value serves as Load ID
   count = 0

   def __init__(self, symbol, value=None):

       '''
       Initializer for the load class

       Inputs
       symbol - String - FX or FY
       value - Float value - Default is None
       '''
       
       #Increment counter
       Load.count += 1

       #Set load id
       self.id = Load.count

       #Set symbol
       self.symbol = symbol

       #Set value
       self.value = value

#Class for Material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the mateelement
        #typecasts user value for E into float
        self.E = float(E)

#Class for Node
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

        #Creating a dictionary to store the dofs
        #belonging to current node
        self.dofs = {}

        #Initializing degrees of freedom for this node
        self.dofs['UX'] = DOF(symbol='UX')
        self.dofs['UY'] = DOF(symbol='UY')

        #Creating a dictionary to store the loads
        #belonging to current node
        self.loads = {}

        #Initializing applied forces at this node
        self.loads['FX'] = Load(symbol='FX')
        self.loads['FY'] = Load(symbol='FY')

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
        col_names = ['Node ID', 'X', 'Y', 'UX', 'UY', 'FX', 'FY']
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
            row.append(str(n.dofs['UX'].value))
            row.append(str(n.dofs['UY'].value))
            row.append(str(n.loads['FX'].value))
            row.append(str(n.loads['FX'].value))
            print(''.join(cell.ljust(cell_width) for cell in row))
        
        #Final print to create space
        print()

#Class for Truss
class Truss():

    def __init__(self):

        #Adding reference to the dictionary of nodes and elements
        self.ndict = Node.ndict
        self.edict = Element.edict


