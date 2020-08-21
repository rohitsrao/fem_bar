#This contains all classes required for preprocessing.

#Creating a class for material
class Material():

    #Initializer
    def __init__(self, name, E):

        #Setting name of the material
        self.name = name

        #Setting the Young's Modulus of the material
        #typecasts user value for E into float
        self.E = float(E)

