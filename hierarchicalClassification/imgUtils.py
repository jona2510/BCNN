"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""
import numpy as np
import os
import random
import albumentations as A
import cv2

# tensorflow
import tensorflow as tf
import tensorflow_hub as hub

def  imageAugmentation(trainSet,cl,H,minI=None,pathSave=None,imgSize = (300,300)):
	# BALANCE WITH RESPECT TO THE LEAVES
	# images withtree chanels
	# minI : integer, minimum number of examples for node
	# only on Train set: https://www.baeldung.com/cs/ml-data-augmentation
	
	#count the number of instances per leaf node
	nin = np.sum(cl,axis=0)	# number of instances per node

	if(minI is None):
		mx = max(nin[H.getLeaves()])	# maximum number of instances	(in a leaf node)
		minI = int(mx * 0.95)	# min number of intances is equal to 90% of the max number of instances		

	nni = 0	# number of new images
	for x in H.getLeaves():
		if(nin[x] < minI):	
			nni += minI - nin[x]

	shTS = np.shape(trainSet)
	shCl = np.shape(cl)
	# create the extended dataset
	newTrain = np.zeros([shTS[0]+nni, shTS[1], shTS[2], shTS[3] ])	# matrix 4-dimensional
	newCl = np.zeros([shCl[0]+nni, shCl[1]],dtype=bool)		# matrix 2-dimensional
	# copy the first images
	newTrain[:shTS[0]] = trainSet
	newCl[:shTS[0]] = cl

	# for reproducible results
	#tf.keras.utils.set_random_seed(0)
	np.random.seed(0)
	random.seed(0)

	#define the pipeline with the tranformations
	transform = A.Compose([
		A.Flip(always_apply=False, p=0.5),
		A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.3, 0.3), rotate_limit=(-180, 180), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
		A.Resize(always_apply=False, p=1.0, height=imgSize[0], width=imgSize[1], interpolation=0)	# corrobrate positions
	])

	#save images
	save = False
	if(pathSave is not None):
		if(os.path.isdir(pathSave)):
			print("Existing dir")
			print("WARNING: Information will be overwritTen!")
			#print("Removing dir and its content...")
			#os.system("rm -r " + pathSave)
		else:
			print("creating dir: ", pathSave)
			os.system("mkdir -p " + pathSave)
			save = True
	

	# add the corresponding images
	cont = 0	
	for x in H.getLeaves():
		imx = np.where(cl[:,x])[0]		# images associated to x
		nimx = len(imx)		# number of instances associated to x
		cx = 0	#
	
		if(save):
			ldir = os.path.join(pathSave,H.nameNodes[x])
			print("creating dir: ", H.nameNodes[x])
			os.system("mkdir " +  ldir)

		dimg = minI - nin[x]
		for i in range(dimg):	#only if dimg is greater than zero
			pos = shTS[0] + cont
			newTrain[ pos ] = transform(image= trainSet[ imx[cx] ] )["image"]
			newCl [ pos ] = H.getSinglePaths()[x]	# corresponding path of labels
			cx = (cx + 1) % nimx	# this helps to iterate multiple times over the orginal set of images
			cont += 1	

			if(save):
				cv2.imwrite( os.path.join( ldir, str(pos) + "_AI.jpg"), newTrain[ pos ])		

	return (newTrain,newCl)

def getFeatures(images, model_handle="https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2", imgSize = (300,300)):
	#return the features of the images
	# efficientnet_v2_imagenet21k_xl return 1280 features for each image
	# images is a set of images that complies imgSize (plus 3 dimensions)
	#
	print("TF version:", tf.__version__)
	print("Hub version:", hub.__version__)
	print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
	tf.keras.utils.set_random_seed(0)	#for reproducible results	#it supposedly do the 3 lines below, but it seems NOT working
	random.seed(0)
	np.random.seed(0)
	tf.random.set_seed(0)

	imgSize = imgSize + (3,)
	if(imgSize != np.shape(images)[1:]):
		raise NameError("Error: Images' size " +str(np.shape(images)[1:])  + " is different of imgSize " +str(imgSize))
	#
	model = tf.keras.Sequential([
		tf.keras.layers.InputLayer(input_shape=imgSize ),	
		hub.KerasLayer(model_handle,
			trainable=False),  # Can be True, see below.
		#tf.keras.layers.Dense(num_classes, activation='softmax')
	])
	model.build()	# got the features
	#model.build((None,) + imgSize)
	model.summary()
	#
	features = model.predict(images)
	del model	#free
	#
	return features
	

