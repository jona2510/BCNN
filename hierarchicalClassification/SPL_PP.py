"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

# Proceses are designed in order to return the 

import numpy as np

def TopDownProcedure(probs,H):
	# check if testSet has the correct dimensions
	shte = np.shape( probs )
	pr = np.zeros((shte[0],H.n),dtype=int) #	[ [] for x in range(len(self.structure)) ]
	#
	lpr = probs 	#self.predict_proba(testSet)		# get all the probabilities of each instance for each node
	#
	# Begin the prediction
	for i in range(shte[0]):	# for each intance
		#print("p: ",i)
		children = H.getRoots()	 #self.H.getChildren()[nl]
		nl = -1	# node with maximmum probability
		while(len(children) > 0):
			pmax = -np.inf	# maximum probability	( restart at each level of hierarchy)
			for x in children:
				##p = self.dCl[x].predict_proba( np.array([testSet[i]]) )
				#p = self.dCl[x].predict_proba( testSet[i].reshape(1,-1) ) 
				p = lpr[i,x]	#p[0,pos_cl[x] ]	# get the probabilities and store the psotive probability
				if(p > pmax):	
					pmax = p
					nl = x
			children = H.getChildren()[nl]	# advance towards the node with highest probability
		pr[i] = H.getSinglePaths()[nl].copy()
	#	
	return pr


def SumOfProbabilities(probs,H):
	# check if testSet has the correct dimensions
	shte = np.shape( probs )
	pred = np.zeros((shte[0],H.n),dtype=bool) #	[ [] for x in range(len(self.structure)) ]

	# for each instance evaluate all the leaf nodes
	sps = np.zeros( (shte[0], len(H.getLeaves()) ))

	for i in range( len(H.getLeaves()) ):
		# path that reaches the i-th leaf
		lp = H.getSetPaths()[ H.getLeaves()[i] ]
		#evaluate all the instances at once
		sps[:,i] = np.average( probs[:, lp ],axis=1 )

	# identify where the max is
	lmax = np.where( np.max( sps,axis=1,keepdims=True) == sps )	# if there is multiple cell with the same max value, it return all of them, only one mus be considered
	#print("lmax: \n",lmax)
	# for i in range 
	c = 0
	#for x,y in zip(lmax[0],lmax[]
	for i in range(len(lmax[0])):
		if(lmax[0][i] == c):
			pred[c] = H.getSinglePaths()[ H.getLeaves()[ lmax[1][i] ] ]	# path of the corresponding leaf
			c += 1

	return pred


class scoreGLB:
	"""
	paper: Hierarchical multilabel classification based on path evaluation 
	Contructor of score Gain Loose Balance (GLB)
	H: object type hierarchy

	# figure 3 (of the paper) shows wrong weights (for levels only adds one to the deepest parent)
	"""
	def __init__(self, H):	

		self.H = H	

		self.levels = np.zeros(self.H.n)
		self.weights = np.zeros(self.H.n)
		self.ML = None		# maximum level

		# estimate levels
		for x in self.H.iteratePF():
			if(x in self.H.getRoots()):		# if it is root
				self.levels[x] = 1.0
			else:
				parents = self.H.getParents()[x]	# the parents of x
				self.levels[x] = np.average( self.levels[parents] ) + 1.0		# average of the parents plus 1
				
		self.ML = max(self.levels)	# maximum level

		#estimate the weigths
		for i in range(self.H.n):
			self.weights[i] = 1 - (self.levels[i] / (self.ML + 1.0))


	def score(self,probs):
		logProbs = np.log(probs)
		shte = np.shape( probs )

		pred = np.zeros((shte[0],self.H.n),dtype=bool) #	[ [] for x in range(len(self.structure)) ]

		# for each instance evaluate all the leaf nodes
		sps = np.zeros( (shte[0], len(self.H.getLeaves()) ))

		for i in self.H.getLeaves():
			# path that reaches the i-th leaf
			lp = self.H.getSetPaths()[i]
			#evaluate all the instances at once
			sps[:,i] = np.sum( logProbs[:,lp]*self.weights[lp] ,axis=1) 
			#sps[:,i] = np.average( probs[:, lp ],axis=1 )

		# identify where the max is
		lmax = np.where( np.max( sps,axis=1,keepdims=True) == sps )	# if there is multiple cell with the same max value, it return all of them, only one mus be considered
		#print("lmax: \n",lmax)
		# for i in range 
		c = 0
		#for x,y in zip(lmax[0],lmax[]
		for i in range(len(lmax[0])):
			if(lmax[0][i] == c):
				pred[c] = self.H.getSinglePaths()[ self.H.getLeaves()[ lmax[1][i] ] ]	# path of the corresponding leaf
				c += 1

		return pred		


	def score2(self,probs):
		"""
			for each path:
				prod_{i=1}{p} w_{i}*P(y_{i})	# prod = multiplications of each node (weight by probability)
			
			applying log:
				sum_{i=1}{p} log(w_{i})+log(P(y_{i}))		# this is the equation used in this function
		"""
		logProbs = np.log(probs)
		shte = np.shape( probs )
		logWeights = np.log(self.weights)

		pred = np.zeros((shte[0],self.H.n),dtype=bool) #	[ [] for x in range(len(self.structure)) ]

		# for each instance evaluate all the leaf nodes
		sps = np.zeros( (shte[0], len(self.H.getLeaves()) ))

		for i in self.H.getLeaves():
			# path that reaches the i-th leaf
			lp = self.H.getSetPaths()[i]
			#evaluate all the instances at once
			sps[:,i] = np.sum( logProbs[:,lp] + logWeights[lp] ,axis=1) 
			#sps[:,i] = np.average( probs[:, lp ],axis=1 )

		# identify where the max is
		lmax = np.where( np.max( sps,axis=1,keepdims=True) == sps )	# if there is multiple cell with the same max value, it return all of them, only one mus be considered
		#print("lmax: \n",lmax)
		# for i in range 
		c = 0
		#for x,y in zip(lmax[0],lmax[]
		for i in range(len(lmax[0])):
			if(lmax[0][i] == c):
				pred[c] = self.H.getSinglePaths()[ self.H.getLeaves()[ lmax[1][i] ] ]	# path of the corresponding leaf
				c += 1

		return pred		








