"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

import sys
import numpy as np
from pyhugin87 import *
from hierarchicalClassification.augmented import probsND
from hierarchicalClassification.imgUtils import imageAugmentation
import copy
from sklearn.metrics import confusion_matrix as cm
from sklearn.ensemble import RandomForestClassifier as rfc
import random
#class weight
from class_weight import generate_class_weights as gcw
# tensorflow:
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa

class BCNN:	
	"""
		Bayesian and Convulutiona Neural Networks (BCNN)
	"""

	def __init__(self, hierarchy, smooth=0.1, model_name="efficientnetv2-xl-21k", BATCH_SIZE=64,epochs=40, weighted=False, search_thr=True, nameNet=None, augmentation=False, pathAugmentation=None):
		"""
		baseClassifier: must be a multilabel classifier, that is, predict_proba returns the probability of being associated to the corresponding label
		hierarchy: is a hierarchy object that contains the hierarchy
		"""

		self.H = hierarchy
		#self.baseClassifier = baseClassifier
		self.domain = None
		self.smooth = smooth
		self.nameNet = nameNet	
		self.search_thr = search_thr	# for estimation of q nodes CPTS; search for the threshold to positively label instances
		# trainWithVal: use only when few data available, it may at least duplicate the training timme
		self.trainWithVal = False	# concatenate training with validation before for the last training	# increase training time...
		#CNN parameters:
		self.model_name = model_name
		self.BATCH_SIZE = BATCH_SIZE
		self.do_fine_tuning = False	#return to false
		self.epochs = epochs # epochs	#it was named ep
		self.weighted = weighted
		self.imgSize = (300,300) + (3,)
		print("TF version:", tf.__version__)
		print("Hub version:", hub.__version__)
		print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
		# IMAGE AUGMENTATION:
		self.augmentation = augmentation
		self.pathAugmentation = pathAugmentation

		model_handle_map = {
			"efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
			"efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
			"efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
			"efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
			"efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
			"efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
			"efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
			"efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
			"efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
			"efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
			"efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
			"efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
			"efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
			"efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
			"efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
			"efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
			"efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
			"efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
			"efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
			"efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
			"efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
			"efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
			"efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
			"efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
			"efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
			"efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
			"efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
			"efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
			"efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
			"efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
			"efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
			"bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
			"bit_m-r152x4": "https://tfhub.dev/google/bit/m-r152x4/1",	# added jsp
			"inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",		# updated https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4
			"inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",	#updated"https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
			"resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
			"resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
			"resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
			"resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
			"resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
			"resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
			"nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
			"nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
			"pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
			"mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
			"mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
			"mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
			"mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
			"mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
			"mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
			"mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
		}

		model_image_size_map = {
			"efficientnetv2-s": 384,
			"efficientnetv2-m": 480,
			"efficientnetv2-l": 480,
			"efficientnetv2-b0": 224,
			"efficientnetv2-b1": 240,
			"efficientnetv2-b2": 260,
			"efficientnetv2-b3": 300,
			"efficientnetv2-s-21k": 384,
			"efficientnetv2-m-21k": 480,
			"efficientnetv2-l-21k": 480,
			"efficientnetv2-xl-21k": 512,
			"efficientnetv2-b0-21k": 224,
			"efficientnetv2-b1-21k": 240,
			"efficientnetv2-b2-21k": 260,
			"efficientnetv2-b3-21k": 300,
			"efficientnetv2-s-21k-ft1k": 384,
			"efficientnetv2-m-21k-ft1k": 480,
			"efficientnetv2-l-21k-ft1k": 480,
			"efficientnetv2-xl-21k-ft1k": 512,
			"efficientnetv2-b0-21k-ft1k": 224,
			"efficientnetv2-b1-21k-ft1k": 240,
			"efficientnetv2-b2-21k-ft1k": 260,
			"efficientnetv2-b3-21k-ft1k": 300, 
			"efficientnet_b0": 224,
			"efficientnet_b1": 240,
			"efficientnet_b2": 260,
			"efficientnet_b3": 300,
			"efficientnet_b4": 380,
			"efficientnet_b5": 456,
			"efficientnet_b6": 528,
			"efficientnet_b7": 600,
			"bit_m-r152x4":  300,	# to search real value
			"inception_v3": 299,
			"inception_resnet_v2": 299,
			"nasnet_large": 331,
			"pnasnet_large": 331,
		}

		self.model_handle = model_handle_map.get(model_name)
		#pixels = model_image_size_map.get(model_name, 224)



	def fit(self, trainSet, cl, validation=None, validation_cl=None):

		#generates the Bayesian Network based on the hierarchy
		self.domain = Domain()

		self.nodes = [None for i in range(self.H.n)]		# references to 'upper' nodes
		self.tables = [None for i in range(self.H.n)]	# references to 'upper' tables

		self.nodesB = [None for i in range(self.H.n)]		# references to 'bottom' nodes; they receive the predictions/probabilities of the classifier
		self.tablesB = [None for i in range(self.H.n)]	# references to 'bottom' tables

		self.valuesAtts = [np.array([False,True]) for i in range(self.H.n)]	# may it be [0,1] ???

		# create root node
		self.nodeR = Node(self.domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.BOOLEAN)	# now it is BOOLEAN, it was NUMBER
		self.nodeR.set_name("root")		
		self.nodeR.set_label("root")		
		# create cpt of root node
		self.tableR = self.nodeR.get_table()
		self.tableR.set_data([0.001,0.999])	# false: 0.001; true: 0.999	

		for x in self.H.iteratePF():	# IMPORTANT: iteration parents first
			# create boolean node
			self.nodes[x] = Node(self.domain, CATEGORY.CHANCE, KIND.DISCRETE, SUBTYPE.BOOLEAN)	# now it is BOOLEAN, it was NUMBER
			#print("name: ",self.H.nameNodes[x])
			self.nodes[x].set_name("N_"+self.H.nameNodes[x])		
			self.nodes[x].set_label("N_"+self.H.nameNodes[x])		
			
			# Probs estimation
			if(x in self.H.getRoots()):	# if x is 'root' node 				
				nz =  len(cl) + 2.0*self.smooth	# normalize
				tr = (len(np.where(cl[:,x] == True )[0]) + self.smooth)/nz	#prob of instances associated to x
				fa = (len(cl) - tr + self.smooth)/nz		#prob of instances NOT associated to x

				# add root as parent
				self.nodes[x].add_parent(self.nodeR)

				# add CPT
				self.tables[x] = self.nodes[x].get_table()
				self.tables[x].set_data([1.0,0.0,fa,tr])	# 00,01,10,11
			else:				
				#add parents to the BN
				parents = self.H.getParents()[x]	# x's parents
				#if(len(parents)>0):	# if have parents, add them to the BN
				for y in range(len(parents)-1,-1,-1):	# added in this way to correspond the cpt generated by probsNB
					self.nodes[x].add_parent(self.nodes[ parents[y] ])
				positions = [x]
				positions.extend(self.H.getParents()[x])

				cpt = probsND(self.valuesAtts, positions, self.smooth )
				cpt.estimateProbs(cl)
				#cpt.probabilities
				cpt = self.ND2HE(cpt)	#cpt in correct format			

				# add the corresponding CPT
				self.tables[x] = self.nodes[x].get_table()
				self.tables[x].set_data( cpt.flatten() )		#[0.1, 0.9, 0.7,0.3,0.3,0.7,0.9,0.1,0.3,0.7,0.5,0.5,0.1, 0.9, 0.7,0.3,0.3,0.7,0.9,0.1,0.3,0.7,0.5,0.5])

		# Train classifier
		if((validation is None) or (validation_cl is None)):
			print("Validation set is not propoerly provided")
			validation = trainSet/255.0
			validation_cl = cl
			valNone = True
		else:
			validation = validation/255.0
			valNone = False

		# augmentation
		if(self.augmentation):
			trainSet,cl = imageAugmentation(trainSet,cl,self.H,minI=None,pathSave=self.pathAugmentation,imgSize=self.imgSize[:2])
			self.instancesAfterAug = np.sum(cl,axis=0)

		# kind of normalization
		#print("shTS: ", np.shape(trainSet))		
		#print("trainSet:\n",trainSet[:2])
		trainSet = trainSet/255.0

		#add new function _fit_CNN
		self._fit_CNN(trainSet, cl, validation, validation_cl)

		#got predictions
		# cnn return probs
		valPred = self.model.predict(validation)		# must return True,False probs are not allowed :(


		# NO TRANFORMATION IS CARRIED OUT FOR CONTINUOUS NODE
		"""
		if(self.search_thr):
			# search for best threshold to label positively the instances
			self.best_m = - np.inf
			for th in np.linspace(0.01,0.99,99,endpoint=True):
				metric = tfa.metrics.F1Score(num_classes=self.H.n, threshold=th, average='macro' )	# average None: returns metric for each label
				metric.update_state(validation_cl, valPred)
				result = metric.result()
				if(result.numpy() > self.best_m):
					self.best_m = result.numpy()	#update best eval
					self.positives_threshold = th
		else:
			self.positives_threshold = 0.5
		
	
		# with threshold probs are transformed to [False,True]
		valPred = valPred > self.positives_threshold
		"""
		
		# Add the 'bottom' nodes as CONITNOUS; that is those who receives the  classifiers prediction
		for x in range(self.H.n):
			# create boolean node
			self.nodesB[x] = Node(self.domain, CATEGORY.CHANCE, KIND.CONTINUOUS)	# continuous node
			self.nodesB[x].set_name("N_"+self.H.nameNodes[x]+"_B")		
			self.nodesB[x].set_label("N_"+self.H.nameNodes[x]+"_B")	

			# add parent node
			self.nodesB[x].add_parent(self.nodes[x])

			# estimate parematers for the Condiciona Gaussian distribuction			
			tr = np.where(validation_cl[:,x])[0]			# where it is True
			fa = np.where(validation_cl[:,x]==False)[0]			# where it is False

			self.nodesB[x].set_alpha( np.average( valPred[fa,x] ) ,0)	# mean; for parent = 0
			self.nodesB[x].set_gamma( np.var( valPred[fa,x] ) ,0)	# variance; for parent = 0

			self.nodesB[x].set_alpha( np.average( valPred[tr,x] ) ,1)	# mean; for parent = 1
			self.nodesB[x].set_gamma( np.var( valPred[tr,x] ) ,1)	# variance; for parent = 1

			## add the corresponding CPT
			#self.tablesB[x] = self.nodesB[x].get_table()
			#lcm = cm( validation_cl[x], valPred[x],labels=[False,True]) + self.smooth#,normalize='true')
			#lcm[0] = lcm[0]/np.sum(lcm[0])
			#lcm[1] = lcm[1]/np.sum(lcm[1])
			#print(self.H.nameNodes[x]," (cm):\n",lcm)
			#self.tablesB[x].set_data( lcm.flatten() )

		if(self.nameNet is not None):
			self.domain.save_as_net(self.nameNet)
		self.domain.compile()

		if(self.trainWithVal):
			if(valNone):
				print("WARNING!: trainWthVal was set True but validation set was not properly provided; re-trainig will be not carried out!!")
			else:
				self._fit_CNN(np.concatenate([trainSet,validation]), np.concatenate([cl, validation_cl]), validation, validation_cl)

		self.isfit = True
		# get parameters through validation

		# train the baseClassifier

	def _fit_CNN(self,trainSet,cl, validation, validation_cl):
		"""
			Add-hoc function for training a CNN
			Return the trained model
		"""
		#self.lclassifier = copy.deepcopy(self.baseClassifier)
		#self.lclassifier.fit(trainSet,cl)
		print("Building model with", self.model_handle)
		tf.keras.utils.set_random_seed(0)	#for reproducible results	#it supposedly do the 3 lines below, but it seems NOT working
		random.seed(0)
		np.random.seed(0)
		tf.random.set_seed(0)

		tf.keras.backend.clear_session()

		self.model = tf.keras.Sequential([
			# Explicitly define the input shape so the model can be properly
			# loaded by the TFLiteConverter
			#tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
			tf.keras.layers.InputLayer(input_shape=self.imgSize ),	# galaxies
			hub.KerasLayer(self.model_handle, trainable=self.do_fine_tuning),
			tf.keras.layers.Dropout(rate=0.2),
			#tf.keras.layers.Dense(len(class_names),		
			tf.keras.layers.Dense(self.H.n,		# 7 only galaxies; hierarchical 9
				kernel_regularizer=tf.keras.regularizers.l2(0.0001),activation='sigmoid' )	# I added: activation='softmax' ; then chan ge by 'sigmoid'
		])
		#model.build((None,)+IMAGE_SIZE+(3,))
		self.model.build((None,)+self.imgSize)
		self.model.summary()


		self.model.compile(
			optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
			#loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),		# from_logits=True
			loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),		# from_logits=True
			#loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits = False, alpha = 0.25, gamma = 2.0, reduction = tf.keras.losses.Reduction.NONE),
			#metrics=['accuracy'])	# it was accuracy
			metrics=['accuracy',tf.keras.metrics.BinaryAccuracy(threshold=0.5),tfa.metrics.F1Score(num_classes=self.H.n,average="macro",threshold=0.5)] )

		steps_per_epoch = len(trainSet) // self.BATCH_SIZE
		validation_steps = len(validation) // self.BATCH_SIZE

		if(self.weighted):
			self.class_weights = gcw(cl,multi_class=False, one_hot_encoded=True)

			self.hist = self.model.fit(
				#train_X, train_label,
				trainSet, cl,
				epochs=self.epochs, 
				# delete steps per epoch because we are NOT using a GENERATOR
				#	https://www.kaggle.com/questions-and-answers/227574
				steps_per_epoch=steps_per_epoch,	
				#validation_data=(valid_X, valid_label),
				validation_data=(validation, validation_cl),
				validation_steps=validation_steps,
				class_weight=self.class_weights).history
		else:
			self.hist = self.model.fit(
				#train_X, train_label,
				trainSet, cl,
				epochs=self.epochs, 
				# delete steps per epoch because we are NOT using a GENERATOR
				#	https://www.kaggle.com/questions-and-answers/227574
				steps_per_epoch=steps_per_epoch,
				#validation_data=(valid_X, valid_label),
				validation_data=(validation, validation_cl),
				validation_steps=validation_steps).history


	def predict(self):
		x = 0
		self.checkIfFit()		

	def predict_proba(self, testSet):
		self.checkIfFit()		
		testSet = testSet/255.0

		# the method predict_proba of some multi-label classifiers (such as sklearn random-forest) gives the probability for each value of each label
		#probs = self.lclassifier.predict_proba(testSet)
		# get the probabilities from the cnn
		probs = self.model.predict(testSet)	
		#print("probs:\n",probs)
		
		pres = np.zeros((len(testSet), self.H.n ))		# return the probability of being associated to each label
		#ppos = np.zeros((len(testSet), self.H.n ))
		#add the evidence to the BN
		#	to the bottom nodes
		for i in range(len(testSet)):
			self.nodeR.select_state(1)	# root node
			for j in range(self.H.n):	# for each node 
				# add the evidence for each state; from a sklearn classifier
				#self.nodesB[j].enter_finding(0, probs[j][i,0])
				#self.nodesB[j].enter_finding(1, probs[j][i,1])
				# addd eviendece from a CNN
				#self.nodesB[j].enter_finding(1, probs[i,j])
				#self.nodesB[j].enter_finding(0, 1.0 - probs[i,j])
				# add the evidence for continuous nodes			
				self.nodesB[j].enter_value(probs[i,j])	# direct value from CNN
				
				#ppos[i,j] = probs[j][i,1]
			# propagate evidence
			self.domain.propagate()

			# get the probabilities
			for j in range(self.H.n):	# for each node 
				pres[i,j] = self.nodes[j].get_belief(1)	# get the probability of True

			# remove all the evidence
			self.domain.retract_findings()
		#print("ppos:\n",ppos)

		return pres

	def predict_proba2(self, testSet):	# the bayesian net receives from the classifier 1 or 0, not the probabilities
		self.checkIfFit()		
		testSet = testSet/255.0
		# the method predict_proba of some multi-label classifiers (such as sklearn random-forest) gives the probability for each value of each label
		#probs = self.lclassifier.predict_proba(testSet)
		# get the probabilities from the cnn
		probs = self.model.predict(testSet)	
		#print("probs:\n",probs)
		
		pres = np.zeros((len(testSet), self.H.n ))		# return the probability of being associated to each label
		#ppos = np.zeros((len(testSet), self.H.n ))
		#add the evidence to the BN
		#	to the bottom nodes
		for i in range(len(testSet)):
			self.nodeR.select_state(1)
			for j in range(self.H.n):	# for each node 
				# add the evidence for each state
				# from sk learn classifier:
				#self.nodesB[j].enter_finding(0, probs[j][i,0])
				#self.nodesB[j].enter_finding(1, probs[j][i,1])
				# from tensorflow CCN :
				if(probs[i,j]>self.positives_threshold):
					self.nodesB[j].select_state(1)
				else:
					self.nodesB[j].select_state(0)
				#ppos[i,j] = probs[j][i,1]
			# propagate evidence
			self.domain.propagate()

			# get the probabilities
			for j in range(self.H.n):	# for each node 
				pres[i,j] = self.nodes[j].get_belief(1)	# get the probability of True

			# remove all the evidence
			self.domain.retract_findings()
		#print("ppos:\n",ppos)

		return pres

	def predict_proba3(self, testSet):	# the bayesian net receives from the classifier 1 or 0, not the probabilities
		self.checkIfFit()		
		testSet = testSet/255.0
		# the method predict_proba of some multi-label classifiers (such as sklearn random-forest) gives the probability for each value of each label
		#probs = self.lclassifier.predict_proba(testSet)
		# get the probabilities from the cnn
		probs = self.model.predict(testSet)	
		#print("probs:\n",probs)
		
		pres = np.zeros((len(testSet), self.H.n ))		# return the probability of being associated to each label
		#ppos = np.zeros((len(testSet), self.H.n ))
		#add the evidence to the BN
		#	to the bottom nodes
		for i in range(len(testSet)):
			self.nodeR.select_state(1)
			for j in range(self.H.n):	# for each node 
				# add the evidence for each state
				# from sk learn classifier:
				#self.nodesB[j].enter_finding(0, probs[j][i,0])
				#self.nodesB[j].enter_finding(1, probs[j][i,1])
				# from tensorflow CCN :
				if(probs[i,j]>self.positives_threshold):
					self.nodesB[j].select_state(1)
				#else:
				#	self.nodesB[j].select_state(0)
				#ppos[i,j] = probs[j][i,1]
			# propagate evidence
			self.domain.propagate()

			# get the probabilities
			for j in range(self.H.n):	# for each node 
				pres[i,j] = self.nodes[j].get_belief(1)	# get the probability of True

			# remove all the evidence
			self.domain.retract_findings()
		#print("ppos:\n",ppos)

		return pres

	"""
	Check is the classifiers is already trained, 
	if not, then raises a exeption
	"""
	def checkIfFit(self):
		if(not self.isfit):
			raise NameError("Error!: First you have to train ('fit') the classifier!!")

	def ND2HE(self,probObj):
	# transform the CPT in the probObj (instance of probsND) to the format requiered by HUGIN EXPERT
	#
		def rec(cd,pos,aNB,aHU):
			"""
			cd: integer, current dimension
			pos: nd array of shape (first_dimension_of_aNB,)
			aNB: CPT from probsND
			aHU: CPT for HUGIN
			"""
			#print("cd :",cd)
			#print("sh aNB: ",len(np.shape(aNB)) )
			#print("sh aHU: ",len(np.shape(aHU)) )
			dm = len(np.shape(aNB))
			#print("pos:\n",pos)
			for x in range( len(probObj.variables[ probObj.positions[cd] ]) ):
				pos[cd] = x
				if(cd == dm-1):	# if it is in the last dimension, 
					posHU = np.zeros(dm,dtype=int)	
					posHU[:dm-1] = pos[1:]
					posHU[-1] = pos[0]	# the first dimension in probsND is the last on HUGIN
					if( np.all(pos[1:]) ):	# if all the parents are positives
						#print("p: ",pos,", ",posHU)
						aHU[tuple(posHU)] = aNB[tuple(pos)]
					else:	# if at least one parent is not positive					
						if(pos[0]==0):
							aHU[tuple(posHU)] = 1.0
						else:
							aHU[tuple(posHU)] = 0.0
				else:
					rec(cd+1,pos,aNB,aHU)					
		#
		# the size is different, the first element in probsNB is the last in HUGIN; the rest of elements should match
		nsize = []
		maxd =  len(probObj.positions)	# dimensions of the CPT
		#print("maxd: ",maxd)
		for i in range(1,maxd):
			nsize.append( len( probObj.variables[i] ) )		# the numer of states/values that can take each node
		nsize.append( len( probObj.variables[0] ) )	# at the end the 
		#
		probsHU = np.zeros( nsize )		# all the elements in a vector
		#print("pHU\n",probsHU)
		pos = np.zeros(maxd,dtype=int)
		#	
		for x in range( len(probObj.variables[ probObj.positions[0] ]) ):
			pos[0] = x
			rec(1,pos,probObj.probabilities,probsHU)			
		#
		return	probsHU		
			



