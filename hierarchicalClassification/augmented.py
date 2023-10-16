"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""


#libraries
import numpy as np

"""
Probabilities of A given its parents (any number of parents)
 P(A | Pa(A) )
"""
class probsND:

	# parents is an array which contains the number of values that can take each one
	# the first is A, the last is the Class, the others are the parents of A
	# positions has the original position of the variables/parents in the structure
	def __init__(self, variables, positions, smooth = 0.1):	
		# all this variable has to contain the information of the class
		#    that is, the class mus be seen as another attribute
		#print("variables:")
		#print(variables)
		z= [ len(variables[positions[i]]) for i in range(len(positions)) ]
		#print("z: ")
		#print(z)


		self.probabilities = np.zeros(z)	#np.zeros(parents)	# creates automatically the n-dimentional array		# P( A | Pa(a))
		self.variables = variables.copy()		#has the number of values that each variable can take, receive the full dictionary
		self.positions = positions.copy() #np.zeros( len(parents) ).astype(int)	#it helps to identify the position of the variable in the orginal structure (data)
		self.smooth = smooth

	#data contains the whole data, and at the end  the column of class is concatenated
	def estimateProbs(self,data):
		position = np.zeros( len(self.positions) ).astype(int)

		# counting
		for i in range(len(self.variables[ self.positions[0] ])):
			position[0] = i 
			x = set( np.where(data[:, self.positions[0] ] == self.variables[ self.positions[0] ][i] )[0] )

			if( len(self.positions) > 1 ):
				self.recCount(1, data, x, position)
			

		# estimate probabilities P( A | Pa(A) )
		position = np.zeros( len(self.positions) ).astype(int)

		if(len(self.positions) > 1):
		
			for i in range(len(self.variables[ self.positions[1] ])):	# it begin in the position 1 of the dic, that is, the first parent 
				position[1] = i
				self.recProbs( 1, position) 

		else:
			self.probabilities = self.probabilities / sum( self.probabilities )


	def recProbs(self, index, position):

		if(index < len(self.positions)):	
			for i in range( len(self.variables[ self.positions[index] ]) ):
				position[index] = i
				self.recProbs(index + 1, position)
		else:	#estimate the probabilities, if all the parents have been visited
			s = 0.0
			for i in range( len(self.variables[ self.positions[0] ]) ):
				position[0] = i
				s += self.probabilities[ tuple(position) ]

			for i in range( len(self.variables[ self.positions[0] ]) ):
				position[0] = i
				self.probabilities[ tuple(position) ] = self.probabilities[ tuple(position) ] / s


	def recCount(self, index, data, set_, position):

		for i in range( len(self.variables[ self.positions[index] ]) ):
			position[index] = i 
			if(set_ != set()):
				x = set( np.where(data[:, self.positions[index] ] == self.variables[ self.positions[index] ][i])[0] ) & set_
			else:
				x=set()

			if( len(self.positions) > (index + 1) ):
				self.recCount(index + 1, data, x, position)
			else:
				#it assigns the value
				self.probabilities[ tuple(position) ] = len(x) + self.smooth
		
	# the full instance and at the end concatenated with the class
	def probsInstance(self, instance):	#it return the "probabilities" for each class
		
		p = [ np.where( self.variables[ self.positions[i] ] == instance[ self.positions[i] ] )[0][0] for i in range( len(self.positions) ) ]
		# It must access directly to the probabilitie
		pr = self.probabilities[tuple(p)]
		#print("---Prob: " + str(pr))
		return pr 

			
		

