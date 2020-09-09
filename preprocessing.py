#This contains all classes required for preprocessing.

#Libraries
import numpy as np
import pandas as pd
import sympy as sp

from sympy.solvers.solveset import linsolve

#Class for Bounday Condition
class BC():

    #Defining a counter variable to keep track of number of BC
    #Counter serves as BC id
    count = 0

    #Keeping a list of dof_ids where boundary conditions are applied
    dof_ids = []

    def __init__(self, n_id, comp, value):

        #Increment counter
        BC.count += 1

        #Set BC id
        self.id = BC.count

        #Set node object
        self.n = Node.ndict[n_id]

        #Set BC
        self.n.dofs[comp].value = float(value)

        #Add DOF id to list
        
        BC.dof_ids.append(self.n.dofs[comp].id)

#Class for DOF
class DOF():

    #Defining a counter as class variable
    #Keeps count of the number of elements created
    #Counter serves as dof id
    count = 0

    #Defining a list to keep track of all created dof_ids
    dof_ids = []

    def __init__(self, symbol, value=None):
        '''
        Initializer for the DOF class

        Inputs 
        symbol - String - ux or uy
        value - Float value - By default set to None
        '''

        #Increment counter
        DOF.count += 1

        #Setting DOF id
        self.id = DOF.count

        #Setting the symbol
        self.symbol = symbol

        #Setting the value
        self.value = value

        #Add the dof_id to the list
        DOF.dof_ids.append(self.id)

#Class for Bar Element
class Element():

    #Defining a counter as class variable
    #Keeps count of number of elements created
    #Counter serves as element id
    count = 0

    #Creating a dictionary to store created elements
    edict = {}    

    #Defining number of dofs in element
    num_dofs = 4

    #Initializer
    def __init__(self, n1, n2, A, mat):

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
        self.mat = mat

        #Setting the area of the element
        self.A = A

        #Defining weights and points for gauss integration
        self.gp = [0]
        self.w = [2]

        #Defining an array to store strains at gauss points
        self.eps_gp_arr = np.zeros(shape=(len(self.gp)))

        #Defining a list to store stresses at gauss points
        self.sig_gp_arr = np.zeros(shape=(len(self.gp)))

        #Creating a dictionary to store nodes belonging to an element
        self.n = {}

        #Adding node objects to the dictionary
        self.n[0] = n1
        self.n[1] = n2

        #Creating a list to keep track of dof_ids belonging to an element
        self.dof_ids = []

        #Adding dof_ids to created list
        for n in self.n.values():
            for d in n.dofs.values():
                self.dof_ids.append(d.id)

        #Adding the created element to the edict dictionary
        Element.edict[self.id] = self

        #Computing element geometry
        self.compute_geometry()

        #Compute the global indices of the element
        self.generate_global_indices()

        #Generate the global stiffness matrix
        self.generate_stiffness_matrix()

        #Transform generated stiffness matrix
        self.transform_k_local_global()

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

            #Extracting the material from the row
            mat_name = df.iloc[i]['mat']

            #Extracting the node objects corresponding to node ids
            n1 = Node.ndict[n1_id]
            n2 = Node.ndict[n2_id]

            #Extracting material object using mat_name
            mat = Material.mat_dict[mat_name]

            #Create node by calling initializer
            cls(n1, n2, A, mat)

        #Deleting dataframe after elements have been created
        del df

    @classmethod
    def define_symbolic_variables(cls):
        '''
        This method defines all the symbolic variables needed to generate
        all the quantities in Element.symbolic_quantities_generator()
        '''

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
        U = Tu
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

        #Defining the inverse of the transformation matrix
        cls.T_inv = cls.T.inv()

    @classmethod
    def stiffness_integrand_generator(cls):

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
    def symbolic_quantities_generator(cls):
        '''
        This method generates all the symbolic quantities needed.
        These include the stiffness matrix integrand, dof_vec, strain, stress
        '''

        #Define necessary symbolic variables
        cls.define_symbolic_variables()

        #Generate the integrand for calculating stiffness matrix
        cls.stiffness_integrand_generator()

        #Symbolic element degree of freedom vector
        cls.dof_vec_sym = sp.Matrix([[cls.u1], [cls.u2]])

        #Symbolic strain vector
        cls.eps_sym = cls.Bt*cls.dof_vec_sym

        #Symbolic stress vector
        cls.sig_sym = cls.E*cls.eps_sym

    def compute_axial_displacements(self):
        '''
        This method is basically to compute the axial displacements of the element.
        The degrees of freedom of a node are in global coordinates i.e UX and UY
        The element formulation is with axial displacements u1 and u2 for both nodes.
        Hence we need to apply the transformation u = T_inv*U
        U here corresponds to self.dof_vec
        '''

        #Substitute the angle of the element to get transformation matrix for current element
        T_inv = Element.T_inv.subs(Element.phi, self.theta)

        #Convert from sympy to numpy array
        T_inv = np.asarray(T_inv).astype(np.float64)

        #Compute the axial dispalcement vector by inverse transform
        self.u_axial = np.matmul(T_inv, self.dof_vec)

    def compute_dof_vec(self):
        '''
        Function to compute the degree of freedom vector of the element
        '''

        #Creating a list to store the the dof values
        dof_values_list = []

        #Looping through the nodes
        for n in self.n.values():

            #Append dof value to the list
            dof_values_list.append(n.dofs['UX'].value)
            dof_values_list.append(n.dofs['UY'].value)

        #Converting list to array
        self.dof_vec = np.array(dof_values_list)

        #Reshaping
        dof_vec_shape = (Element.num_dofs, 1)
        self.dof_vec = np.reshape(self.dof_vec, newshape=dof_vec_shape)

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

    def compute_strain(self):
        '''
        This method computes the element strain at all the gauss points
        '''

        #Looping through all gauss points
        for i in range(len(self.gp)):
            
            #Extracting the natural coordinate of the gauss point
            xi_val = self.gp[i]

            #Defining a list of values that will be used for substitution
            sub_list = [(Element.xi, xi_val), (Element.L, self.L),
                        (Element.u1, self.u_axial[0, 0]),
                        (Element.u2, self.u_axial[2, 0])]

            #Substituting in symbolic strain
            eps = Element.eps_sym.subs(sub_list)

            #Converting to numpy
            eps = np.asarray(eps).astype(np.float64)

            #Adding strain value to the list
            self.eps_gp_arr[i] = eps[0, 0]



    def gauss_integrator(self, integrand):
        '''
        Performs gauss integration on integrand
        integrand must be a symbolic expression in xi (symbolic)
        '''

        #Defining a list to store the gauss integrals
        gauss_integrals = []

        #Generating Et_values through yield check
        Et_val_list = self.yield_check()

        #Looping through number of gauss points
        for i in range(len(self.gp)):

            sub_list = [(Element.xi, self.gp[i]), (Element.E, Et_val_list[i])]
            gauss_integral = self.w[i]*integrand.subs(sub_list)
            gauss_integrals.append(gauss_integral)

        #Calculating total integrand
        #Need to provide a start argument for the sum as symbolic matrix of zeros
        integral = sum(gauss_integrals, sp.Matrix(np.zeros(shape=integrand.shape)))

        return integral

    def generate_global_indices(self):
        '''
        This method generates a matrix of tuples which is the same size
        as the local 2d stiffness matrix. Each tuple correponds to the row,col
        in the global stiffness matrix where the corresponding term of the stiffness matrix
        goes
        '''

        #Creating a list to store the indices
        global_indices_list = []

        #Permuting the dof ids to get the global indices
        for i in range(Element.num_dofs):
            for j in range(Element.num_dofs):
                tup = (self.dof_ids[i], self.dof_ids[j])
                global_indices_list.append(tup)

        #Converting list into array
        self.global_indices = np.empty(len(global_indices_list), dtype=object)
        self.global_indices[:] = global_indices_list
        newshape = (Element.num_dofs, Element.num_dofs)
        self.global_indices = np.reshape(self.global_indices, newshape=newshape)

    def generate_reduced_indices(self):
        '''
        This method generates reduced indices for assembly into the reduced stiffness matrix
        '''

        #Making a copy of the global indices
        self.reduced_indices = np.copy(self.global_indices)

        #Looping through the tuples
        for i in range(Element.num_dofs):
            for j in range(Element.num_dofs):

                #Each compoment of the reduced_indices array is a tuple of indices
                #indicate where that element of the local stiffness matrix fits into the
                #global stiffness matrix. Now if any of the dof_ids have a BC applied to them
                #we set the tuple to be (None, None). If no bc is applied to that DOF then its 
                #new position in the reduced global stiffness matrix is computed by subtracting 
                #from that index the number of dof_ids that are lower than that index
                #Setting the reudced indices to None if index is in BC.dof_ids
                if any(x in self.reduced_indices[i, j] for x in BC.dof_ids) == True:

                    self.reduced_indices[i, j] = (None, None)

                else:

                    #Extracting the indices of the tuple
                    idx1 = self.reduced_indices[i, j][0]
                    idx2 = self.reduced_indices[i, j][1]

                    #Figure out how many dof_ids are lesser than idx1 and idx2
                    count1 = len([k for k in BC.dof_ids if k < idx1])
                    count2 = len([k for k in BC.dof_ids if k < idx2])

                    #Updating the index values based on the respective count values
                    idx1 -= count1
                    idx2 -= count2

                    #Inserting the updated values back into reduced_indices matrix
                    self.reduced_indices[i, j] = (idx1, idx2)

    def generate_stiffness_matrix(self):
        '''
        This method is to generate the element stiffness matrix for the 
        1D bar via gauss integration
        '''

        #Compute the symbolic 1d stiffness matrix
        self.k_local_1d_s = self.gauss_integrator(Element.integrand)

        #Convert from symbolic matrix into numpy matrix
        sub_list = [(Element.L, self.L),
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

    def yield_check(self):
        '''
        This function checks if the material has yielded at the gauss points.
        It returns a list of values for Et
        '''

        #Creating an empty list to store the Et values and which will be returned
        Et_val_list = []

        #Looping through the list containing strain values at the gauss points
        for i in range(self.eps_gp_arr.shape[0]):

            #Extracting strain value at a single gauss point
            strain_at_gp = self.eps_gp_arr[i]

            #If material has yielded at gauss point set the tangent modulus to updated value
            if strain_at_gp >= self.mat.yp:

                Et_val_list.append(self.mat.Et(strain_at_gp))

            #else set tangent modulus to be Young's Modulus
            else:

                Et_val_list.append(self.mat.E)

        return Et_val_list

#Class for Load
class Load():

   #Defining a counter as class variable
   #Keeps count of number of loads created
   #counter value serves as Load ID
   count = 0
   def __init__(self, symbol, value=0.0):
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
       self.value = float(value)

#Class for Material
class Material():

    #Defining a dictionary to store all created material objects
    mat_dict = {}

    #Initializer
    def __init__(self, name, E, **kwargs):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the mateelement
        #typecasts user value for E into float
        self.E = float(E)

        #Adding material object to dictionary
        Material.mat_dict[self.name] = self

        #If **kwargs dictionary is not empty
        if kwargs:

            #Storing the stress strain polynomial in the plastic region
            #as sig_poly and also calculating the expression for Tangent
            #Modulus as the derivative of sig_poly. Please note sig_poly
            #is a function of nominal values and is a function of the 
            #total strain and not just plastic strain
            if 'sig_poly' in kwargs.keys():

                self.sig_poly = kwargs['sig_poly']

                #Computing tangent modulus
                self.Et = self.sig_poly.deriv()

            #Storing the yield point
            if 'yp' in kwargs.keys():

                self.yp = kwargs['yp']

    @classmethod
    def display_material_data(cls):

        #Looping through all the materials created
        for mat in cls.mat_dict.values():

            print('Material: {}'.format(mat.name))
            print('--------------------------------------------------------------')
            print('Young\'s Modulus: {}'.format(mat.E))

            if hasattr(mat, 'yp'):
                print('Yield point: {}'.format(mat.yp))

            print()

            if hasattr(mat, 'sig_poly'):
                print('Stress strain polynomial in plastic region: ')
                print(mat.sig_poly)
                print()
                print('Tangent Modulus: ')
                print(mat.Et)
                print()

#Class for Node
class Node():

    #Defining a counter as class variable
    #Keeps count of number of nodes created
    #Counter value serves as node id
    count = 0

    #Creating a dictionary to store created nodes
    #keys are the node ids
    ndict = {}

    #Number of DOFs per node
    num_dofs = 2

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

        #Creating a dictionary to store loads with dof_id
        #being the keys
        self.loads_by_dof_ids = {}

        #Initializing applied forces at this node
        self.loads['FX'] = Load(symbol='FX')
        self.loads['FY'] = Load(symbol='FY')

        #Adding to the loads_by_dof_ids dictionary
        self.loads_by_dof_ids[self.dofs['UX'].id] = self.loads['FX']
        self.loads_by_dof_ids[self.dofs['UY'].id] = self.loads['FY']

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
            row.append('{:.4f}'.format(n.x))
            row.append('{:.4f}'.format(n.y))

            #For UX and UY we need to check if the value is None or not 
            #before formatting its ouput

            if n.dofs['UX'].value == None:
                row.append(str(n.dofs['UX'].value))
            else:
                row.append('{:.4f}'.format(n.dofs['UX'].value))

            if n.dofs['UY'].value == None:
                row.append(str(n.dofs['UY'].value))
            else:
                row.append('{:.4f}'.format(n.dofs['UY'].value))

            row.append('{:.2f}'.format(n.loads['FX'].value))
            row.append('{:.2f}'.format(n.loads['FY'].value))

            print(''.join(cell.ljust(cell_width) for cell in row))
        
        #Final print to create space
        print()

    def apply_load(self, comp, value):
        '''
        Applies the load component at a node
        comp is a string
        '''
        self.loads[comp].value = float(value)

    def apply_load_by_dof_id(self, dof_id, value):
        '''
        sets value for the load at the nodes but uses
        dof id to index them
        '''
        self.loads_by_dof_ids[dof_id].value = value

#Class for Truss
class Truss():

    def __init__(self):

        #Adding reference to the dictionary of nodes and elements
        self.ndict = Node.ndict
        self.edict = Element.edict

        #Defining the dimension of the global stiffness matrix for the truss
        self.global_dimension = len(Node.ndict)*Node.num_dofs
        
    def assemble_full_stiffness(self):
        '''
        Assembles the global stiffness matrix for the truss
        '''

        #Define a matrix of zeros
        k_shape = (self.global_dimension, self.global_dimension)
        self.K = np.zeros(shape=k_shape)

        #Loop through each element
        for e in self.edict.values():

            #Looping through the local stiffness matrix
            for i in range(e.num_dofs):
                for j in range(e.num_dofs):

                    #Extracting the global indices
                    gi = e.global_indices[i, j][0]
                    gj = e.global_indices[i, j][1]

                    #Subtracting 1 from global indices as 
                    #indexings starts from 0 
                    gi -= 1
                    gj -= 1

                    self.K[gi, gj] += e.k_local_2d[i, j]

    def assemble_reduced_stiffness(self):
        '''
        This method assembles the reduced stiffness matrix after the application of 
        boundary conditions
        '''

        #Initializing the reduced stiffness matrix
        reduced_shape = (self.reduced_dimension, self.reduced_dimension)
        self.Kr = np.zeros(shape=(reduced_shape))

        #Looping through each element
        for e in self.edict.values():

            #Compute the reduced indices for the element
            e.generate_reduced_indices()

            #Loop through the element stiffness matrix and place each 
            #element appropriately into the reduced stiffness matrix using the 
            #reduced indices. Skip if the reduced indices are (None, None)
            for i in range(Element.num_dofs):
                for j in range(Element.num_dofs):

                    #If the reduced indices are not (None, None)
                    if None not in e.reduced_indices[i, j]:

                        #Extract the global indices
                        gi = e.reduced_indices[i, j][0]
                        gj = e.reduced_indices[i, j][1]

                        #Subtracting 1 from the indices as element indexing starts from zero
                        #But tuples contain indices starting from
                        gi -= 1
                        gj -= 1

                        #Adding to reduced stiffness matrix
                        self.Kr[gi, gj] += e.k_local_2d[i, j]
        
    def apply_bcs_from_csv(self, f):
        '''
        This method applies boundary conditions from a csv file
        csv flie must contain columns - node id, comp, value
        comp must be a string 'UX' or 'UY'
        '''

        #Creating a dataframe from csv
        df = pd.read_csv(f)

        #Computing the number of rows in the dataframe
        num_rows = df.shape[0]

        #Loop through the rows
        for i in range(num_rows):

            #Extract the node id, component and value
            n_id = df.iloc[i]['node id']
            comp = df.iloc[i]['comp']
            value = df.iloc[i]['value']

            #Apply BC
            BC(n_id, comp, value)

        #Save a reference to the list of nodes with BC in Truss object
        self.bc_dof_ids = BC.dof_ids

        #Compute the reduced dimension
        self.compute_reduced_dimension()

        #Compute active dofs
        self.compute_active_dofs()

    def apply_loads_from_csv(self, f):
        '''
        This method applies loads at nodes from csv

        Inputs
        f - csv - columns should be node id, comp, value
        comp stands for component
        '''

        #Creating a dataframe from csv
        df = pd.read_csv(f)

        #Compute the number of rows in the csv
        num_rows = df.shape[0]

        #Loop through the rows
        for i in range(num_rows): 

            #Extract the node id, component and value
            n_id = df.iloc[i]['node id']
            comp = df.iloc[i]['comp']
            value = df.iloc[i]['value']

            #Extract the node object
            n = self.ndict[n_id]

            #Apply load
            n.apply_load(comp, value)

        #Deleting dataframe
        del df

    def apply_residue_to_nodes(self, res_vec):
        '''
        Needed for NewtonRaphson solver. It takes the components
        of the residual vector and applies them to the appropriate
        nodal dofs

        Inputs
        res_vec - residual vector
        '''

        #Looping through the nodes
        for n in self.ndict.values():

            #Looping through the dofs
            for d in n.dofs.values():

                #If the dof is an active dof then update load value
                #from residue vector. 
                #The row for the residue vector will be the index in
                #self.active_dofs where the dof_id appears.
                if d.id in self.active_dofs:

                    row = self.active_dofs.index(d.id)
                    n.apply_load_by_dof_id(d.id, res_vec[row, 0])

    def compute_active_dofs(self):
        '''
        Generating a list of active dofs. Active dof_ids are those dofs where no boundary
        condition has been specified. 
        '''

        self.active_dofs = [i for i in DOF.dof_ids if i not in BC.dof_ids] 

    def compute_reduced_dimension(self):
        '''
        Function computes the dimension of the reduced stiffness matrix and load vector
        '''

        #Compute the dimension of the reduced stiffness matrix
        self.reduced_dimension = self.global_dimension - len(self.bc_dof_ids)

    def generate_reduced_force_vec(self):
        '''
        This method generates the reduced force vector
        '''
        
        #Creating a list to store the forces applied on the DOFs without boundary condition
        reduced_force_list = []

        #Looping through the nodes
        for n in self.ndict.values():

            #Looping through the nodal degrees of freedom
            for dof in n.dofs.values():

                #Checking if DOF is free from boundary conditions
                if dof.id not in BC.dof_ids:

                    #Append forces to the list
                    if dof.symbol == 'UX':
                        reduced_force_list.append(n.loads['FX'].value)
                    if dof.symbol == 'UY':
                        reduced_force_list.append(n.loads['FY'].value)

        #Converting list to an array
        self.Fr = np.array(reduced_force_list)

        #Reshaping
        Fr_shape = (self.reduced_dimension, 1)
        self.Fr = np.reshape(self.Fr, newshape=Fr_shape)

    def prep_for_solving(self):
        '''
        This method is to compute some basic parameters and initiliaze variables
        needed for solving
        '''

        #Initialize a reduced displacement vector with zeros to store the values
        #of displacment vector at the active dofs
        u_shape = (self.reduced_dimension, 1)
        self.u = np.zeros(shape=u_shape)


    def solve_elastic(self):
        '''
        This method solves the reduced stiffness matrix and reduced force vector
        '''

        #Solving the system to compute the displacement increment
        self.du = np.linalg.solve(self.Kr, self.Fr)

        #Updating the reduced displacement vector
        self.u += self.du

    def update_dofs(self):
        '''
        This method is to update the dof values after solve_elastic() has successfully run
        '''

        #Looping through the nodes
        for n in self.ndict.values():

            #Looping through the dofs in the node
            for dof in n.dofs.values():

                #If the dof is an active dof
                if dof.id in self.active_dofs:

                    #Then set the value of that dof to be the value from the 
                    #corresponding value from the solution vector. The index of the
                    #corresponding term in the solution vector is determined from the 
                    #dof_id's index in self.active_dofs
                    row = self.active_dofs.index(dof.id)
                    dof.value = self.u[row, 0]



