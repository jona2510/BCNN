"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np 
from collections import deque
from scipy.io import arff
import re
from sklearn import preprocessing

class loadHdata:

	def __init__(self,typeFile="hcbncc"):
		self.typeFile = typeFile

	#load head (hierarchy) used in BNCC method
	def loadBNCCheader(self,headerFile):	
		dic = {}
		cl_wop = ""
		cl_names = ""
		with open(headerFile,"r") as f:
			ban=True
			while(ban):
				line=f.readline()
				line = line.lower()
				spl=line.split("class")
				if(len(spl)>1):
					cl_wop=spl[1].strip()		#obtiene las clases sin procesamiento, se usara para sacar la estructura del DAG

				spl=line.split("@orden")
				if(len(spl)>1):				
					spl2=spl[1].strip().split(",")
					cl_names = spl2
					c=0
					for i in range(0,len(spl2)):
						dic[spl2[i].strip()]=c
						c=c+1
					ban=False
		f.close()
		
		nn=len(dic)		#numeros de nodos
		mest=np.zeros((nn,nn),dtype=int)	#matriz que representa la estructura del arbol/poliarbol/grafo

		spl=cl_wop.split(",")
		for i in range(0,len(spl)):
			spl2=spl[i].split("/")
			if(spl2[0].strip()!="root"):
				mest[ dic[spl2[0].strip()], dic[spl2[1].strip()] ] = 1
			else:
				print("no se agrego root,*")
		return (cl_names, mest)


	#load data used in BNCC method		
	def loadBNCCdata(self,dataFile):
		data=[]
		classes=[]
		with open(dataFile,"r") as f:
			for line in f:
				spl=line.strip().split(";")		# attributes and classes are separated by a semicolon		

				#data.append(",".join(spl))
				data.append(spl[0])		#agrego directaamente los atributos
				classes.append( np.array(spl[1].split(","),dtype=int) )	#agrego las clases como cadenas de ceros y unos
		f.close()
		classes = np.concatenate([classes])	# matrix of shape(n_instances,m_labels)

		ndata=len(data)
		natt=len(data[0].split(","))	#numero de atributos	
		datafc=np.zeros((ndata,natt))	# only numeric attributes

		for i in range(0,ndata):
			daux=data[i].split(",")
			for j in range(0,natt):
				#comprueba que no sea dato faltante
				if(daux[j]=="?" or daux[j]==""):
					datafc[i,j]==np.nan
				else:
					datafc[i,j]=float(daux[j])

		return (datafc,classes)

	#
	def saveArff(self,name,structure,data,labels,dataTest=None,labelsTest=None,label_names=None,att_names=None):
		"""
		name: 			str, 
		structure: 		ndarray of shape (n_labels, n_labels), the hierarchy
		data:			ndarray of shape (m_instances, a_atts), the dataset
		dataTest:		ndarray of shape (m2_instances, a_atts), the test dataset
		label_names:	list of shape (n_labels,), name of labels
		att_names:		list of shape (a_atts,) name of attributes
		"""

		shS = np.shape(structure)
		shD = np.shape(data)
		shL = np.shape(labels)

		if(shL[0] != shD[0]):
			raise NameError("Error, number size of label different to data")

		if( shS[0] == shS[1] ):
			if( not(label_names is None) ):
				shLN = np.shape(label_names)
				if(shLN[0] != shS[0]):
					raise NameError("Error, label_names and strcuture different sizes")
			else:	# auto create name labels
				label_names = [ ("label_" + str(i)) for i in range(shS[0])]
		else:
			raise NameError("Error, structure")

		if(shL[1] != shS[0]):
			raise  NameError("Error, number of labels in 'labels' is different to size of 'structure'")

		if( not(att_names is None)):
			shAN = np.shape(att_names)
			if(shAN[0] != shD[1]):
				raise NameError("Error, number of att_names and atts on data are different")
		else:		#auto create name atts
			att_names = [ ("att_" + str(i)) for i in range(shD[1])]

		if( not(dataTest is None)):
			shTe = np.shape(dataTest)			
			if(shTe[1] != shD[1]):
				raise NameError("Error, number of atts on data and dataTest are different")			

		s = "@RELATION 'my_relation: -C "+ str(shS[0]) +"'\n"

		#if(label_names is None):	# auto name labels
		#	for i in range(shS[0]):
		#		s = s + "\n@ATTRIBUTE label_" + str(i) + " {0,1}"
		#else:
		for x in label_names:
			s = s + "\n@ATTRIBUTE " + x + " {0,1}"

		#if(att_names is None):	# auto name atts
		#	for i in range(shD[1]):
		#		s = s + "\n@ATTRIBUTE att_" + str(i) + " numeric"
		#else:
		for x in att_names:
			s = s + "\n@ATTRIBUTE " + x + " numeric"

		s = s+ "\n\n@DATA"

		# save data
		labels = labels.astype(int)
		with open(name+".arff","w") as f:
			f.write(s)
			for x,y in zip(data,labels):
				f.write("\n" + ",".join( y.astype(str) ) + "," + ",".join( x.astype(str) ) )
		f.close()

		# save dataTest
		if(not(dataTest is None)):
			with open(name+"_test.arff","w") as f:
				f.write(s)
				for x,y in zip(dataTest,labelsTest):
					f.write("\n" + ",".join( y.astype(str) ) + "," + ",".join( x.astype(str) ) )
			f.close()

		# save hierarchy
		h = hierarchy(structure)
		with open(name+".hierarchy","w") as f:
			parents = h.getParents()
			roots = h.getRoots()
			flag = True
			for x in h.iteratePF():		# this iteration guarantee that a 'root' will be first written in the file
				if(x in roots):
					if(flag):	# first time not print '\n'
						flag = False
					else:
						f.write("\n")
					f.write("root." + label_names[x])
				else:
					#for each parents
					for y in parents[x]:
						f.write("\n" + label_names[y] + "." + label_names[x])											
		f.close()


	def loadArff(self,name):
		n = -np.inf
		with open(name,"r") as f:
			line = f.readline()
			line = line.lower()
			if(re.search('@relation',line)):
				spl = line.split("-c")
				if(len(spl) == 2):
					aux = re.sub(r'[^\w\s]',"",spl[1])
					n = int(aux)
					if(n < 1):
						raise NameError("Error!, number of classes can not be lower than 1")
				else:
					raise NameError("Error!, file '" + name + "' do not have number of classes, p.e. '-C 5'")
			else:
				raise NameError("Error!, The first line has to contain @RELATION ")
		f.close()

		data,meta = arff.loadarff(name)
		l = [x for x in meta]

		# return data, labels, att names, label names, 
		return ( np.array(data[ l[n:] ].tolist(),dtype=float) , np.array(data[ l[:n] ].tolist(),dtype=int) , l[n:], l[:n])
				

	def loadHierachyArff(self,nameH,label_names):
		n = len(label_names)
		if(n<1):
			raise NameError("Error!, label_names is a list with the name of labels")
		st = np.zeros((n,n),dtype=bool)

		dic_inv = {}	
		c=0
		for x in label_names:
			dic_inv[x] = c
			c = c + 1

		with open(nameH,"r") as f:
			for line in f:
				line = line.strip()
				if(len(line) > 0):
					spl = line.split(".")
					if(len(spl)==2):
						if(spl[0] != 'root'):
							st[ dic_inv[spl[0]], dic_inv[spl[1]] ] = True	# spl[0] is parent of spl[1]
					elif(len(spl)<2):
						raise NameError("There is no relation parent.child: "+ line)							
					else:
						raise NameError("There are too many '.', only one '.' per line is allowed: " + line)

		return st		

	def loadHMC(self, name, atts2float=True):
		dic={}
		dic_inv={}	
		dinv = []	# initially, it has the same values than dic_inv but in an array way
		cl_wop=""
		atts = []	# name of the attributes
		atts_type = []	# type of the attrbute: 1 if it is nominal 0 if it is numeric 
		atts_vals = []	# values of the attributes
		hdata = []
		with open(name,"r") as f:
			ban=True
			while(ban):
				line = f.readline()

				#spl=line.split("class")
				spl = line.split()
				#if(len(spl)>1):
				if("class" in  spl):
					#print("encontre class")
					spl2=spl[-1].strip().split(",")
					cl_wop=spl[-1].strip()		#obtiene las clases sin procesamiento, se usara para sacar la estructura del DAG
					c=0
					#print("len(spl2):")
					#print(len(spl2))
					for i in range(0,len(spl2)):
						spl3=spl2[i].split("/")
						for j in range(0,len(spl3)):
							if((spl3[j]!="root") and (not spl3[j] in dic)):
								dic[spl3[j]]=c
								dic_inv[c]=spl3[j]
								dinv.append(spl3[j])
								c=c+1
							#else:
							#	print(spl3[j])
					ban=False
					#line=spl[0]+"class {"+spl2[1]+"}\n"		
				else: 			#save attributes
					if((line.lower().find("@attribute")) >= 0):	# save the attributte
						spl2 = line.split()
						atts.append(spl2[1])	#only the name of the attribute
						
						if((line.lower().find("numeric")) >= 0):	# if it is numeric then 0, else it is nominal (1)
							atts_type.append(False)	# it is numeric
							atts_vals.append( spl2[-1] )
						else:
							atts_type.append(True)	# it is nominal		
							a_aux = spl2[-1].replace("{","")
							a_aux = a_aux.replace("}","")
							atts_vals.append( a_aux.split(",") )	# delete { }
							
			#get the data
			ban = True
			while( ban ):
				line = f.readline()
				#spl = line.split()
				if((line.lower().find("@data")) >= 0):
					ban = False
			#hdata = f.readline()
			line = f.readline()
			while( line ):	# saves the data
				hdata.append(line.split(","))	# they are split by comma
				line = f.readline()
		#print("atts:\n",atts)
		#exit()

		f.close()
		nn=len(dic)		#numeros de nodos
		#print("numero de etiquetas: "+str(nn))
		nd=(np.zeros(nn)).astype(int)	#numero de instancias por nodo
		mest=(np.zeros((nn,nn))).astype(int)	#matriz que representa la estructura del arbol/poliarbol/grafo

		#obtiene la estructura del DAG
		spl=cl_wop.split(",")
		for i in range(0,len(spl)):
			spl2=spl[i].split("/")
			if(spl2[0]!="root"):
				mest[dic[spl2[0] ],dic[spl2[1] ]]=1
			else:
				print("no se agrego root,*")

		atts_type = np.array(atts_type)		# ndarray 0: numeric, 1:nominal
	
		


		#create hierarchy
		H = hierarchy(mest)
		H.initialize()

		# process the classes to binary vectors
		ninst = len(hdata)
		fclasses = np.zeros( (ninst, nn) ,dtype=bool)		# here, the full paths will be saved
		for i in range( ninst ):
			mnl = []
			#for x in hdata[i,-1].split("@")	# get the last node of each path to which the instances is associated
			for x in hdata[i].pop(-1).strip().split("@"):	# get the last node of each path to which the instances is associated
				mnl.append( dic[x] )
			fclasses[i] = H.combinePaths(mnl)

		hdata = np.array(hdata,dtype=object)
		#"""
		for x in np.where(atts_type)[0]:	# for each nominal att. check if there is '?'
			print("nominal: "+str(x), " :: " ,set(hdata[:,x]))
			if(np.any( np.where(hdata[:,x]=='?')[0] )):
				#add an extra value to the categories
				#print("adding to: ", atts_vals[x])
				atts_vals[x].append('?')
		#"""
		
		#atts_vals[np.where(atts_type)[0][2]].append('?')	# for seq_GO insteaf of upper block 
		#atts_vals[np.where(atts_type)[0][3]].append('?')	# for seq_GO

		atts_vals = np.array(atts_vals,dtype=object)		# a way to acces easier to each value

		if(not atts2float):		# if true, then transform the data to float (this cause trouble if the attributes are not numeric)
			# creates the values that each
			print("inicio*/*/*/*/*/**/*/*/*//*/*/*/")			 
			print("values of nominal attributes:\n",atts_vals[atts_type])			
			print("fin-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n")
			#enc = preprocessing.OneHotEncoder(categories=atts_vals[atts_type],drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			enc = preprocessing.OneHotEncoder(categories=atts_vals[atts_type].tolist(),drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			#enc = preprocessing.OneHotEncoder(drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			print("hdata_nominals:")
			print(hdata[:,atts_type],"\n")
			enc.fit(hdata[:,atts_type])
			print("inicio*/*/*/*/*/**/*/*/*//*/*/*/")			 
			print("(enc) values of nominal attributes:\n",enc.categories_)			
			print("fin-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n")

			n_nom = np.sum(atts_type)		# number of nominal attributes
			n_num = len(atts_type) - n_nom	# number of numeric attributes

			n_bin = 0	# number of 'binary' attributes after transforming all the nominal attributes
			for x in atts_vals[atts_type]:
				lx = len(x)
				if(lx > 2):
					n_bin += lx
				elif( lx == 2):
					n_bin += 1
				else:
					raise NameError("ERROR: nominal attrbiute has lower than 2 values: ",x)

			hdata_aux = np.zeros((len(hdata), n_num+n_bin),dtype=object)
			hdata_aux[:,:n_num] = hdata[:, np.logical_not(atts_type) ]#.astype(float)	# copy the numerical atts
			hdata_aux[:,n_num:] = enc.transform( hdata[:,atts_type] ).toarray()	# transform the nominal data

			del hdata
			hdata = hdata_aux.astype(str)
	
			# modify 'atts': the name of the attributes
			atts = np.array(atts)
			atts_aux = np.zeros(n_num+n_bin,dtype=object)
			atts_aux[:n_num] = atts[np.logical_not(atts_type)]

			c = n_num
			for x,y in zip(atts_vals[atts_type],atts[atts_type]):			
				lx = len(x)
				if(lx > 2):
					#n_bin += lx
					for z in x:
						atts_aux[c] = y+"__"+z+"_mod"
						c += 1						
				else:
					#n_bin += 1
					atts_aux[c] = y+"__"+x[0]+"-0_"+x[1]+"-1_mod"
					c += 1

			atts = atts_aux

			
		else:
			hdata = hdata.astype(float)
			#hdata = np.array(hdata)

		# return data, labels, att names, label names, structure 
		return (hdata, fclasses, atts, dinv, mest)


	def loadHMC_tree(self, name, atts2float=True):
		#load original FunCat datasets but return them taking into account only the first path to which instances are associated

		dic={}
		dic_inv={}	
		dinv = []	# initially, it has the same values than dic_inv but in an array way
		cl_wop=""
		atts = []	# name of the attributes
		atts_type = []	# type of the attrbute: 1 if it is nominal 0 if it is numeric 
		atts_vals = []	# values of the attributes
		hdata = []
		with open(name,"r") as f:
			ban=True
			while(ban):
				line = f.readline()

				#spl=line.split("class")
				spl = line.split()
				#if(len(spl)>1):
				if("class" in  spl):
					#print("encontre class")
					spl2=spl[-1].replace("/","_").strip().split(",")	
					cl_wop=spl[-1].replace("/","_").strip()		#obtiene las clases sin procesamiento, se usara para sacar la estructura del DAG
					c=0
					#print("len(spl2):")
					#print(len(spl2))
					for i in range(0,len(spl2)):
						#spl3=spl2[i].split("/")
						#for j in range(0,len(spl3)):
						if((spl2[i] != "root") and (not spl2[i] in dic)):
							dic[spl2[i]] = c
							dic_inv[c] = spl2[i]
							dinv.append(spl2[i])
							c=c+1
							#else:
							#	print(spl3[j])
					ban=False
					#line=spl[0]+"class {"+spl2[1]+"}\n"		
				else: 			#save attributes
					if((line.lower().find("@attribute")) >= 0):	# save the attributte
						spl2 = line.split()
						atts.append(spl2[1])	#only the name of the attribute
						
						if((line.lower().find("numeric")) >= 0):	# if it is numeric then 0, else it is nominal (1)
							atts_type.append(False)	# it is numeric
							atts_vals.append( spl2[-1] )
						else:
							atts_type.append(True)	# it is nominal		
							a_aux = spl2[-1].replace("{","")
							a_aux = a_aux.replace("}","")
							atts_vals.append( a_aux.split(",") )	# delete { }
							
			#get the data
			ban = True
			while( ban ):
				line = f.readline()
				#spl = line.split()
				if((line.lower().find("@data")) >= 0):
					ban = False
			#hdata = f.readline()
			line = f.readline()
			while( line ):	# saves the data
				hdata.append(line.split(","))	# they are split by comma
				line = f.readline()
		#print("atts:\n",atts)
		#exit()

		f.close()
		nn=len(dic)		#numeros de nodos
		#print("numero de etiquetas: "+str(nn))
		nd=(np.zeros(nn)).astype(int)	#numero de instancias por nodo
		mest=(np.zeros((nn,nn))).astype(int)	#matriz que representa la estructura del arbol/poliarbol/grafo


		for x in dic.keys():
			#cada x indica una tryaectoria
			spl=x.split("_")
			pl=spl[0]
			for i in range(1,len(spl)):	
				pla=pl+"_"+spl[i]			
				mest[dic[pl],dic[pla]] = 1
				pl=pla
			

		##obtiene la estructura del DAG
		#spl=cl_wop.split(",")
		#for i in range(0,len(spl)):
		#	spl2=spl[i].split("/")
		#	if(spl2[0]!="root"):
		#		mest[dic[spl2[0] ],dic[spl2[1] ]]=1
		#	else:
		#		print("no se agrego root,*")

		atts_type = np.array(atts_type)		# ndarray 0: numeric, 1:nominal
	
		


		#create hierarchy
		H = hierarchy(mest)
		H.initialize()

		# process the classes to binary vectors
		ninst = len(hdata)
		fclasses = np.zeros( (ninst, nn) ,dtype=bool)		# here, the full paths will be saved
		for i in range( ninst ):
			mnl = []
			#for x in hdata[i,-1].split("@")	# get the last node of each path to which the instances is associated
			#for x in hdata[i].pop(-1).strip().split("@"):	# get the last node of each path to which the instances is associated
			#	mnl.append( dic[x] )
			#fclasses[i] = H.combinePaths(mnl)
			fclasses[i] = H.getSinglePaths()[ dic[ hdata[i].pop(-1).strip().split("@")[0].replace("/","_") ] ]

		hdata = np.array(hdata,dtype=object)
		#"""
		for x in np.where(atts_type)[0]:	# for each nominal att. check if there is '?'
			print("nominal: "+str(x), " :: " ,set(hdata[:,x]))
			if(np.any( np.where(hdata[:,x]=='?')[0] )):
				#add an extra value to the categories
				#print("adding to: ", atts_vals[x])
				atts_vals[x].append('?')
		#"""
		
		#atts_vals[np.where(atts_type)[0][2]].append('?')	# for seq_GO insteaf of upper block 
		#atts_vals[np.where(atts_type)[0][3]].append('?')	# for seq_GO

		atts_vals = np.array(atts_vals,dtype=object)		# a way to acces easier to each value

		if(not atts2float):		# if true, then transform the data to float (this cause trouble if the attributes are not numeric)
			# creates the values that each
			print("inicio*/*/*/*/*/**/*/*/*//*/*/*/")			 
			print("values of nominal attributes:\n",atts_vals[atts_type])			
			print("fin-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n")
			#enc = preprocessing.OneHotEncoder(categories=atts_vals[atts_type],drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			enc = preprocessing.OneHotEncoder(categories=atts_vals[atts_type].tolist(),drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			#enc = preprocessing.OneHotEncoder(drop='if_binary',handle_unknown='error')	# ,sparse_output=False
			print("hdata_nominals:")
			print(hdata[:,atts_type],"\n")
			if( np.any(atts_type) ):
				print("atts:type: ",atts_type)
				enc.fit(hdata[:,atts_type])
				print("inicio*/*/*/*/*/**/*/*/*//*/*/*/")			 
				print("(enc) values of nominal attributes:\n",enc.categories_)			
				print("fin-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n\n")

				n_nom = np.sum(atts_type)		# number of nominal attributes
				n_num = len(atts_type) - n_nom	# number of numeric attributes

				n_bin = 0	# number of 'binary' attributes after transforming all the nominal attributes
				for x in atts_vals[atts_type]:
					lx = len(x)
					if(lx > 2):
						n_bin += lx
					elif( lx == 2):
						n_bin += 1
					else:
						raise NameError("ERROR: nominal attrbiute has lower than 2 values: ",x)

				hdata_aux = np.zeros((len(hdata), n_num+n_bin),dtype=object)
				hdata_aux[:,:n_num] = hdata[:, np.logical_not(atts_type) ]#.astype(float)	# copy the numerical atts
				hdata_aux[:,n_num:] = enc.transform( hdata[:,atts_type] ).toarray()	# transform the nominal data

				del hdata
				hdata = hdata_aux.astype(str)
	
				# modify 'atts': the name of the attributes
				atts = np.array(atts)
				atts_aux = np.zeros(n_num+n_bin,dtype=object)
				atts_aux[:n_num] = atts[np.logical_not(atts_type)]

				c = n_num
				for x,y in zip(atts_vals[atts_type],atts[atts_type]):			
					lx = len(x)
					if(lx > 2):
						#n_bin += lx
						for z in x:
							atts_aux[c] = y+"__"+z+"_mod"
							c += 1						
					else:
						#n_bin += 1
						atts_aux[c] = y+"__"+x[0]+"-0_"+x[1]+"-1_mod"
						c += 1

				atts = atts_aux
			#else:
			#	hdata = hdata.astype(float)
			
		else:
			hdata = hdata.astype(float)
			#hdata = np.array(hdata)

		# return data, labels, att names, label names, structure 
		return (hdata, fclasses, atts, dinv, mest)


		
class policiesLCN:

	"""
	constructor of policies
	hierarchy : hierarchy object
	"""
	def __init__(self,H,balanced=False,policy="siblings"):
		self.H = H	#hierarchy object
		self.balanced = balanced
		self.policy = policy
		#self.seed = seed

	def getInstances(self,node,dataset,clSet):
		if(self.policy == "siblings"):
			return self.getSiblings(node,dataset,clSet)
		elif(self.policy == "inclusive"):
			return self.inclusive(node,dataset,clSet)
		elif(self.policy == "lessInclusive"):
			return self.lessInclusive(node,dataset,clSet)
		elif(self.policy == "balancedBU"):	#	balanced Bottom-Up
			return self.balancedBU(node,dataset,clSet)
		elif(self.policy == "redSiblings"):	#	redudant siblings
			return self.getSiblings(node,dataset,clSet)
		else:
			raise NameError("The requested policy does not existe: ",self.policy)

	"""
	FIRST VALIDATE THAT ALL NODES HAVE SIBLINGS
	siblings policy

	node: int, node with the positive classes
	dataset: ndarray of shape (x,y), where x is the number of instances and y the number of attributes
	clSet: ndarrya of shape (x,c), where x is the number of instances and c the number of labels/nodes
	balanced: if True, return the same number of negative intances as positives (delete negative or positive instances if needed)
	"""
	def getSiblings(self,node,dataset,clSet):
		shd = np.shape(dataset)
		shc = np.shape(clSet)
		#validate dataset and clSet
		self.validate(node,shd,shc)
		indP = np.where(clSet[:,node] == 1 )[0]	# indices of positive instances
		#print("positives: \n",indP)
		if(len(indP)==0):
			raise NameError("there are not instances associated to node: ",node)

		sibs = self.H.getSiblings()
		indN = set()	# indices of negative instances
		for x in sibs[node]:
			indN = indN | set( np.where( (clSet[:,node] == 0) & (clSet[:,x] == 1) )[0] )	# instances associated to siblings of x, but not associated to x
		indN = np.array(sorted(indN))
		#print("negatives: \n",indN)
		# commented so can return only positives
		if(len(indN)==0):
			#indN = set( [np.where( clSet[:,node] == 0 )[0][0]] )
			indNa = np.where( clSet[:,node] == 0 )[0] 
			#raise NameError("there are not instances associated to sibblings but not-associated to node: ",node)
			if(len(indNa)==0):
				raise NameError("there are not negative instances for node: ",node)			
			else:
				indN = np.array([ indNa[0] ])
				print("WARNING!!: there are not instances associated to sibblings but not-associated to node: ",node,", one random negative instance is returned")

		if(self.balanced):
			indP,indN = self.banlancePN(indP,indN)
		#print(node,": ",len(indP),", ",len(indN))
		ind = np.array( sorted( set(indP) | set(indN) ) )	# indices instances to be returned
		#d  = dataset[ind]
		#cl = clSet[ind]
		return ind #(d.copy(),cl[:,node].copy())

	"""
	FIRST VALIDATE THAT ALL NODES HAVE SIBLINGS
	Redundant Siblings policy:
		- it search positives in node and node descendants
		- it search negatives in siblings and siblings descendants
		
	node: int, node with the positive classes
	dataset: ndarray of shape (x,y), where x is the number of instances and y the number of attributes
	clSet: ndarrya of shape (x,c), where x is the number of instances and c the number of labels/nodes
	balanced: if True, return the same number of negative intances as positives (delete negative or positive instances if needed)
	"""
	def getRedSiblings(self,node,dataset,clSet):
		shd = np.shape(dataset)
		shc = np.shape(clSet)
		#validate dataset and clSet
		self.validate(node,shd,shc)

		indP = set(np.where(clSet[:,node] == 1 )[0])	# indices of positive instances
		for x in self.H.getDescendants()[node]:		# positives of descendants are also considered
			indP = indP | set( np.where( clSet[:,x] == 1 )[0] )	# instances associated to descendats of x
		
		#print("positives: \n",indP)
		if(len(indP)==0):
			raise NameError("there are not instances associated to node: ",node)

		sibs = self.H.getSiblings()
		sibD = set(sibs[node])	# siblings and siblings' descendats of node 
		indN = set()	# indices of negative instances

		for x in sibs[node]:
			sibD |= self.H.getDescendants()[x]	

		for x in sibD:
			indN = indN | set( np.where( (clSet[:,x] == 1) )[0] )	# instances associated to siblings of x,

		#remove from indN instances in indP
		indN = indN - indP

		#print("negatives: \n",indN)
		if(len(indN)==0):
			#indN = set( [np.where( clSet[:,node] == 0 )[0][0]] )
			indNa = set(np.where( clSet[:,node] == 0 )[0] ) - indP
			#raise NameError("there are not instances associated to sibblings but not-associated to node: ",node)
			if(len(indNa)==0):
				raise NameError("there are not negative instances for node: ",node)			
			else:
				indNa = np.array(sorted(indNa))
				indN = np.array([ indNa[0] ])
				print("WARNING!!: there are not instances associated to {sibblings and siblings descendats} but not-associated to {node and node descendants}: ",node,", one random negative instance is returned")

		indP = np.array(sorted(indP))
		indN = np.array(sorted(indN))

		if(self.balanced):
			indP,indN = self.banlancePN(indP,indN)
		print(node,": ",len(indP),", ",len(indN))
		ind = np.array( sorted( set(indP) | set(indN) ) )	# indices instances to be returned
		#d  = dataset[ind]
		#cl = clSet[ind]
		return ind #(d.copy(),cl[:,node].copy())

	"""
	less inclusive policy
		return all the instances (highly unbalance for deep nodes), unless banlanced = True

	node: int, node with the positive classes
	dataset: ndarray of shape (x,y), where x is the number of instances and y the number of attributes
	clSet: ndarrya of shape (x,c), where x is the number of instances and c the number of labels/nodes
	balanced: if True, return the same number of negative intances as positives (delete negative or positive instances if needed)
	"""
	def lessInclusive(self,node,dataset,clSet):
		shd = np.shape(dataset)
		shc = np.shape(clSet)
		#validate dataset and clSet
		self.validate(node,shd,shc)
		indP = np.where(clSet[:,node] == 1 )[0]	# indices of positive instances
		#print("positives: \n",indP)
		if(len(indP)==0):
			raise NameError("there are not instances associated to node: ",node)

		indN = np.where( clSet[:,node] == 0 )[0]		# instances not associated to x
		if(len(indN)==0):
			raise NameError("there are not negative instances for node: ",node)			

		if(self.balanced):
			indP,indN = self.banlancePN(indP,indN)
		print(node,": ",len(indP),", ",len(indN))
		ind = np.array( sorted( set(indP) | set(indN) ) )	# indices instances to be returned
		#d  = dataset[ind]
		#cl = clSet[ind]
		return ind


	"""
	inclusive policy

	node: int, node with the positive classes
	dataset: ndarray of shape (x,y), where x is the number of instances and y the number of attributes
	clSet: ndarrya of shape (x,c), where x is the number of instances and c the number of labels/nodes
	balanced: if True, return the same number of negative intances as positives (delete negative or positive instances if needed)
	"""
	def inclusive(self,node,dataset,clSet):
		shd = np.shape(dataset)
		shc = np.shape(clSet)
		#validate dataset and clSet
		self.validate(node,shd,shc)
		indP = np.where(clSet[:,node] == 1 )[0]	# indices of positive instances
		#print("positives: \n",indP)
		if(len(indP)==0):
			raise NameError("there are not instances associated to node: ",node)

		roots = self.H.getRoots()
		ancs = self.H.getAncestors()[node]
		#dif = set(roots) - set(ancs)	# roots that not are ancestors of node

		indN = set(np.where( clSet[:,node] == 0 )[0])	# indices of negative instances
		#for x in dif:
		#	indN = indN | set( np.where( clSet[:,x] == 1 )[0] )	# instances no-associated to ancestors of x

		#this function is more complex than previous, but it is uselful for 'non-consistent' paths
		for x in ancs:
			indN = indN & set( np.where( clSet[:,x] == 0 )[0] )	# instances no-associated to ancestors of x

		indN = np.array(sorted(indN))

		if(len(indN) == 0):
			#indN = set( [np.where( clSet[:,node] == 0 )[0][0]] )		
			indNa = np.where( clSet[:,node] == 0 )[0] 
			#raise NameError("there are not instances associated to sibblings but not-associated to node: ",node)
			if(len(indNa)==0):
				raise NameError("there are not negative instances for node: ",node)			
			else:
				indN = np.array([indNa[0]])
				print("WARNING!!: there are not instances no-associated to acestors of node: ",node,", one random negative instance is returned")			

		if(self.balanced):
			indP,indN = self.banlancePN(indP,indN)
		ind = np.array( sorted( set(indP) | set(indN) ) )	# indices instances to be returned
		#d  = dataset[ind]
		#cl = clSet[ind]
		return ind


	def balancePN(self,iP,iN):
		lp = len(iP)
		ln = len(iN)
		if(lp > ln):			
			iP = iP[:ln]	# remove the last instances
		elif(ln > lp):
			iN = iN[:lp]	# remove the last instances
		return (iP,iN)

	"""
	balancedBU, balanced Bottom-Up, 

	node: int, node with the positive classes
	dataset: ndarray of shape (x,y), where x is the number of instances and y the number of attributes
	clSet: ndarrya of shape (x,c), where x is the number of instances and c the number of labels/nodes
	balanced: if True, return the same number of negative intances as positives (delete negative or positive instances if needed)
	"""
	def balancedBU(self,node,dataset,clSet):
		shd = np.shape(dataset)
		shc = np.shape(clSet)
		#validate dataset and clSet
		self.validate(node,shd,shc)
		vp = (clSet[:,node] == 1)	# vector of positives
		indP = np.where( vp )[0]	# indices of positive instances
		#vp = ~vp # negate vp	(useful below)
		vp = (clSet[:,node] == 0)	# vector of negatives 
		maxins = len(indP)	# maximum number of instances

		#print("positives: \n",indP)
		if(len(indP)==0):
			#raise NameError("there are not instances associated to node: ",node)
			print("EXTREME WARNING!! there are not POSITIVE instances for node: ",node," a negative instance will be returned" )
			maxins = 1

		roots = self.H.getRoots()
		ancs = self.H.getAncestors()[node].copy()
		depths = self.H.getDepths()
		#dif = set(roots) - set(ancs)	# roots that not are ancestors of node

		if(len(ancs)==0):	# nodes without ancestors are at depth 1
			#indN = np.where( clSet[:,node] == 0 )[0]	# indices of negative instances
			indN = np.where( vp )[0]	# indices of negative instances												
			if(len(indN)>maxins):
				indN = indN[:maxins]	
			indN = set(indN)
		else:
			md = depths[node]	# node's depth 
			d = md - 1
			indN = set()
			while(d>0):				
				# combine negative instances in the same depth
				vl = np.zeros(shd[0],dtype=bool)	# a vector full of False
				for x in np.where(depths[ancs] == d)[0]:
					vl = vl | ((clSet[:, ancs[x] ] == 1) & vp )		# associated to ancestor of node, but not associated to node

				# select those instances that are NOT in set of negative instances
				indnl = set(np.where(vl)[0])	
				indnl = list(indnl - indN)	# keep instances in indnl but not in indN

				# add negative instances up to maxins
				lil = len(indnl)
				if( lil > 0):	# if there are negative instances to add
					toadd = maxins - len(indN)
					indN = indN | ( set( indnl[:toadd] ) )

				if(len(indN) == maxins):
					break	# break the cycle
				d-=1

			if( (d == 0) and (len(indN) < maxins) ):	# all negative instances
				# select those instances that are NOT in set of negative instances
				indnl = set(np.where(vp)[0])	
				indnl = list(indnl - indN)	# keep instances in indnl but not in indN

				# add negative instances up to maxins
				lil = len(indnl)
				if( lil > 0):	# if there are negative instances to add
					toadd = maxins - len(indN)
					indN = indN | ( set( indnl[:toadd] ) )				

		if(len(indN) == 0):
			#indN = set( [np.where( clSet[:,node] == 0 )[0][0]] )		
			indNa = np.where( clSet[:,node] == 0 )[0] 
			#raise NameError("there are not instances associated to sibblings but not-associated to node: ",node)
			if(len(indNa)==0):
				print("EXTREME WARNING!! there are not negative instances for node: ",node )
				#raise NameError("there are not negative instances for node: ",node)			
			else:
				indN = np.array([indNa[0]])
				print("WARNING!!: there are not instances no-associated to acestors of node: ",node,", one random negative instance is returned")			
		print(node,": ",len(indP),", ",len(indN))
		ind = np.array( sorted( set(indP) | set(indN) ) )	# indices instances to be returned
		#d  = dataset[ind]
		#cl = clSet[ind]
		return ind
		

	def validate(self,node,shDataset,shClSet):
		if(len(shDataset)!=2):
			raise NameError("Error!!, datasets has to be a ndarray of shape (x,y), where x is the number of instances and y the number of attributes")
		if(len(shClSet)!=2):
			raise NameError("Error!!, clSet has to be a ndarray of shape (x,c), where x is the number of instances and c the number of labels/nodes")
		if(shDataset[0] != shClSet[0]):
			raise NameError("Error!!, The number of instances in datasets and clSet is different: ",shDataset[0],", ",shClSet[0])
		if(not (0 <= node < shClSet[1]) ):
			raise NameError("Error!!, node out of limits")


class hierarchy:


	"""
	Contructor of hierachy
	
	matrix: is a nd array of shape (n,n), where n is the number of nodos (excluding the root node)
	"""
	def __init__(self, structure,nameNodes=None):	
		x = 1
		shm = np.shape(structure)
		if(len(shm) == 2):
			if(shm[0] != shm[1]):
				raise NameError("Error, structure has to be a ndarray of shape (n,n), where n is the number of nodos (excluding the root node)")
		else:
			raise NameError("Error, structure has to be a ndarray of shape (n,n), where n is the number of nodos (excluding the root node)")

		if(nameNodes is None):
				self.nameNodes = [str(i) for i in range(shm[0])]	# the name of the node is equal to its position in the structure
		else:
			if(len(nameNodes)!=shm[0]):
				raise NameError("Error, nameLabels' size has to be equal to one of the dimensions of structure")
			else:
				self.nameNodes = nameNodes	

		self.n = shm[0]			# the number of labels
		self.structure = structure.copy()	# the cell [i,j] has 1 if the i-th node is parent of the j-th node, 0 otherwise
		self.roots = None		# children of root node, thie fist level of the hierarchy
		self.leaves = None		# leaf nodes;  root and leaf nodes are not exclusives
		self.parents = None		# list with parents of each node
		self.children = None	# list with children of each node
		self.siblings = None	# list with siblings of each node
		self.ancestors = None	# list with ancestors of each node
		self.descendants = None	# list with descendants of each node
		self.LCN = None			# list with classifiers of each node, (Local) (C)lassifiers per (N)ode approach.
		self.paths = None		# list with the "path" to reach a node (full nd array) (contains the "same information" than ancestors)
		self.orderPF = None		# list with the order in which can be iterated ALL the nodes in (P)arents (F)irst fashion, this guarantee that the parents of a node were already visited
		self.orderCF = None		# list with the order in which can be iterated ALL the nodes in (C)hildren (F)irst fashion, this guarantee that the children of a node were already visited
		self.setPaths = None
		self.depths = None		# Depths of each node (works for DAG's)
		self.internals = None	# internal nodes



	"""
	vector (is a ndarray of size (n,)) has to be valid path (has to fulfill the hierarchical constraint)
	So, 'getDephestNodes' returns the 'leaves' of the graph/hierarchy formed by vector
		work for DAG's
		This function may be improved by a BPP (for GRAPS)
	"""
	def getDephestNodes(self,vector):
		p = np.where(vector)[0]	# positives
		dn = []
		for x in p:	# for each positive label
			ch = self.getChildren()[x]	#get children of x
			fl = True
			for y in ch:	# check if children of x are in p
				if(y in p):	# if child is in p, then x is not a dephest node 
					fl = False					
					break
			if(fl):
				dn.append(x)
		return np.array(dn)

	"""
	Return the "roots"/first-level of the hierarchy
	"""
	def getRoots(self):
		if( self.roots is None ):
			self.roots = np.where(sum( self.structure ) == 0)[0]			
		return self.roots

	"""
	Return the leaf nodes of the hierarchy
	"""
	def getLeaves(self):
		if( self.leaves is None ):
			self.leaves = np.where(sum( self.structure.transpose() ) == 0)[0]
		return self.leaves

	"""
	Return the parent(s) of each node
	"""
	def getParents(self):
		if( self.parents is None ):
			self.parents = np.empty(self.n, dtype = object)
			for i in range(self.n):
				self.parents[i] = np.where(self.structure[:,i] == 1)[0]				
		return self.parents

	"""
	Return the children of each node
	"""
	def getChildren(self):
		if( self.children is None ):
			self.children = np.empty(self.n, dtype = object)
			for i in range(self.n):
				self.children[i] = np.where(self.structure[i] == 1)[0]				
		return self.children

	"""
	Return the siblings of each node
	"""
	def getSiblings(self):
		if( self.siblings is None ):
			self.siblings = np.empty(self.n, dtype = object)
			roots = self.getRoots()
			
			parents = self.getParents()
			children = self.getChildren()
			for i in range(self.n):
				if(i in roots):
					#for x in roots:
					#	self.siblings[x] = np.array(sorted( set(roots) - set([x])))	# only siblings
					self.siblings[i] = np.array(sorted( set(roots) - set([i])))	# only siblings		
				else:
					sib = set()
					for j in parents[i]:
						sib = sib | set(children[j])
					self.siblings[i] = np.array(sorted( sib - set([i])))	# only siblings	
		return self.siblings

	"""
	Return the ancestors of each node
	"""
	def getAncestors(self):
		def getAncestors_rec(node,ancestors,visited,parents):		
			nodeAnc = []
			for x in parents[node]:		# Parent ancestors are ancestors of node
				nodeAnc.append(x)		# Parent is ancestor
				if( not visited[x] ):	# if parent has not assigned ancestors yet, first get its ancestors
					getAncestors_rec(x,ancestors,visited,parents)

				for y in ancestors[x]:	# Parent ancestors
					if(y not in nodeAnc):
						nodeAnc.append(y)
			ancestors[node] = np.array(sorted(nodeAnc))
			visited[node] = True
			

		if( self.ancestors is None ):
			self.ancestors = np.empty(self.n, dtype = object)
			roots = self.getRoots()
			parents = self.getParents()
			visited = np.zeros(self.n,dtype=bool)	# all is false

			for x in roots:	
				self.ancestors[x] = np.array([]).astype(int)	# roots do not have parents
				visited[x] = True

			for i in range(self.n):
				if(not visited[i]):	
					getAncestors_rec(i,self.ancestors,visited,parents)
		return self.ancestors

	"""
	Return the descendants of each node
	"""
	def getDescendants(self):
		def getDescendants_rec(node,descendants,visited,children):		
			nodeDes = []
			for x in children[node]:		# Children descendants are decendent of node
				nodeDes.append(x)		# Child is descendent
				if( not visited[x] ):	# if child has not assigned descendents yet, first get its descendants
					getDescendants_rec(x,descendants,visited,children)

				for y in descendants[x]:	# Child descendants
					if(y not in nodeDes):
						nodeDes.append(y)
			descendants[node] = np.array(sorted(nodeDes))
			visited[node] = True
			
		if( self.descendants is None ):
			self.descendants = np.empty(self.n, dtype = object)
			leaves = self.getLeaves()
			children = self.getChildren()
			visited = np.zeros(self.n,dtype=bool)	# all is false

			for x in leaves:	
				self.descendants[x] = np.array([]).astype(int)	# leaves do not have children
				visited[x] = True

			for i in range(self.n):
				if(not visited[i]):	
					getDescendants_rec(i,self.descendants,visited,children)
		return self.descendants

	"""
	get the paths to reach each node, vector format
	"""
	def getSinglePaths(self):
		if( self.paths is None ):
			self.paths = np.zeros((self.n,self.n),dtype=bool)
			anc = self.getAncestors()
			for i in range(self.n):
				for j in anc[i]:
					self.paths[i,j] = True
				self.paths[i,i] = True		# the path considers itself
		return self.paths

	"""
	Return the depths of each node
	Note for DAG's:
		the depth of a node with multiple parents is the depthest parent plus one
	"""
	def getDepths(self):
		if(self.depths is None):
			self.depths = np.zeros(self.n,dtype=int)
			parents = self.getParents()
			for x in self.iteratePF():
				if(len(parents[x])==0):
					self.depths[x] = 1
				else:
					indp = parents[x][ np.where( self.depths[ parents[x] ] == max(self.depths[ parents[x] ]) )[0][0] ]	# index of the deepest parent
					self.depths[x] = self.depths[ indp ] + 1
		return self.depths


	"""
	All nodes that are not leaves are returned as internal nodes
	"""
	def getInternals(self):
		if(self.internals is None):
			self.internals = np.array( sorted(set([i for i in range(self.n)]) - set(self.getLeaves() ) ) )
		return self.internals

	"""
	get the list of nodes that form the path to reach each a node, list format
	"""
	def getSetPaths(self):
		if( self.setPaths is None ):
			self.setPaths = np.empty(self.n, dtype = object)	#np.zeros((self.n,self.n),dtype=bool)
			anc = self.getAncestors()
			for i in range(self.n):				
				#for j in anc[i]:
				#	self.paths[i,j] = True
				#self.paths[i,i] = True		# the path considers itself
				self.setPaths[i] = np.zeros( len(anc[i]) + 1 ,dtype=int)	# same number of ancestor plus one
				self.setPaths[i][:-1] = anc[i]	#copy ancestor
				self.setPaths[i][-1] = i			# add itself 
				self.setPaths[i] = np.sort(self.setPaths[i])	#sort
		return self.setPaths

	"""
	combine the paths on 'nodes' to generate a 'multilabel path'
	"""
	def combinePaths(self,nodes):
		ml = np.zeros(self.n,dtype=bool)
		if(len(nodes)==0):
			print("Warning: nodes has zeros elements, returning a vector full of zeros")
		else:			
			for x in set(nodes):
				ml = self.getSinglePaths()[x] | ml
		return ml


	"""
	Force the vectors to be consisnt with the hierarchy adding ancestors of nodes
	vectors: ndarray (dtype=bool) of shape(m_instances,n_nodes)
	"""
	def forceConsistency(self,vectors,overwrite=False):
		shv = np.shape(vectors)
		if(len(shv) != 2):
			raise NameError("ERROR: vectors has to be a ndarray of shape(m_instances,n_nodes)")
		if(shv[1] != self.n):
			raise NameError("ERROR: vectors has to be a ndarray of shape(m_instances,n_nodes) with n_nodes equal to: ",self.n)			

		if(overwrite):
			vcon = vectors
		else:
			vcon = np.zeros(shv,dtype=bool)

		for x in self.iteratePF():
			f = np.where( vectors[:,x] )[0]		# instances associated to x
			if(len(f)>0): 
				vcon[f] = vcon[f] | self.getSinglePaths()[x]

		return vcon


	"""
	run most of the methods in orde to have the information 'available'
	"""
	def initialize(self):
		self.getRoots()
		self.getLeaves()
		self.getParents()
		self.getChildren()
		self.getSiblings()
		self.getAncestors()
		self.getDescendants()
		self.getSinglePaths()
		self.getSetPaths()
		
	"""
	iterate (P)arents (F)irst
	return a list with the order in which can be iterated ALL the nodes in a TD fashion, 
		this guarantee that the parents of a node were first visited.
	"""
	def iteratePF(self):
		if(self.orderPF is None):
			#roots = []
			queue = deque()	# the queue
			visited = []	# list with the visited nodes
			#visited = np.zeros(len(self.structure)).astype(bool)
		
			for i in self.getRoots():
				queue.append(i)
				#this gets the parents of the i-th class, which form the chain of the i-th node/class		
				#self.dChain[i] = set( np.where( self.structure[:,i] == 1 )[0] )	# a set is better option

			# generate the order for training the classifiers, and then predict
			while( len(queue) > 0 ):
				y = queue.popleft()		#get the node/class
		
				flag = True
				#firts checks if all its parents are already visited
				#for x in self.dChain[y]:
				for x in (self.getParents()[y]):
					if( x not in visited ):	
						flag = False
						break

				if(flag):
					#if all the parents are visited
					# the node/class is visited
					visited.append(y)
					# and add its children at the end of the queue	(only those that haven't been visited and aren't in the queue)
					children = self.getChildren()[y]		#np.where( self.structure[y] == 1 )[0]
				
					for x in children:
						if( (x not in visited) and (x not in queue) ):
							queue.append(x)				
				else:				
					#if SOME parent has NOT been visited
					# insert the node/class in the end of the queue 
					queue.append(y)	

			if( ( len(set(visited)) != self.n ) or ( len(visited) != self.n )):
				raise NameError("ERROR in the construction of the order for trainig the classifiers (review code)")

			# the order to build the classifier is saved in visited
			self.orderPF = visited
		return self.orderPF

	"""
	iterate (C)hildren (F)irst
	return a list with the order in which can be iterated ALL the nodes in a TD fashion, 
		this guarantee that the parents of a node were first visited.
	"""
	def iterateCF(self):
		if(self.orderCF is None):
			self.orderCF = np.flip( self.iteratePF() )
		return self.orderCF



if __name__ == "__main__":
	
	sDAG= np.zeros((9,9),dtype=int)
	sDAG[0,2] = 1
	sDAG[0,3] = 1
	sDAG[1,3] = 1
	sDAG[1,4] = 1
	sDAG[1,5] = 1
	sDAG[3,6] = 1
	sDAG[4,6] = 1
	sDAG[4,7] = 1
	sDAG[6,8] = 1

	h = hierarchy(sDAG)
	print("roots: \n",h.getRoots())
	print("\nleaves: \n",h.getLeaves())
	print("\nparents:")
	p = h.getParents()
	for i in range(h.n):
		print(i,": ",p[i])
	print("\nchildren: ")
	p = h.getChildren()
	for i in range(h.n):
		print(i,": ",p[i])
	print("\nsiblings: ")
	p = h.getSiblings()
	for i in range(h.n):
		print(i,": ",p[i])
	print("\nancestors: ")
	p = h.getAncestors()
	for i in range(h.n):
		print(i,": ",p[i])
	print("\ndescendants: ")
	p = h.getDescendants()
	for i in range(h.n):
		print(i,": ",p[i])
	print("\nPaths: ")
	p = h.getSinglePaths()
	for i in range(h.n):
		print(i,": ",p[i],", ", np.where(p[i]==True)[0])
	p = h.getSetPaths()
	for i in range(h.n):
		print(i,": ",p[i])

	print("\nml prediction [2,7]:")
	p = h.combinePaths([2,7])
	print(p,", ", np.where(p==True)[0])

	print("\nml prediction [3,7]:")
	p = h.combinePaths([3,7])
	print(p,", ", np.where(p==True)[0])

	print("\nml prediction [2,5,7,8]:")
	p = h.combinePaths([2,5,7,8])
	print(p,", ", np.where(p==True)[0])

	print("\nIterate Parents first: \n",h.iteratePF())
