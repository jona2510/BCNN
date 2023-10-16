import numpy as np
from hierarchicalClassification.hStructure import loadHdata as ldata
import matplotlib.pyplot as plt2
from keras.utils import to_categorical
import os
import tensorflow as tf
from hierarchicalClassification.hStructure import hierarchy
from sklearn.model_selection import StratifiedShuffleSplit	# stratified split
#preprocesing images:
import cv2
import imutils
# image aumentation
import albumentations as A


def img_shape(path,pathD):
	# pre-processing for images, only resize them to 300,300
	# img_shape("RYDLS-20/Pneumocystis/","RYDLS-20_pp/Pneumocystis/")	#	COVID-19/      MERS/ Normal/	Pneumocystis/	SARS/	Streptococcus/	Varicella/
	dic = {}	# key is the shape
	#
	print("Reading images from: ",path)
	#
	cant = 0
	lpath = path	#os.path.join(imgpath, str(i))
	jpg = 0
	png = 0
	transform = A.Compose([	
		A.Resize(always_apply=False, p=1.0, height=300, width=300, interpolation=0)
	])
	with os.scandir( lpath ) as ficheros:
		#print("len ficheros: ", len(ficheros))
		for filename in ficheros:
			#ficheros = [fichero.name for fichero in ficheros if fichero.is_file() and fichero.name.endswith('.jpg')]
			#if( filename.is_file() ):
			#print("fn: ",filename)
			#if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename.name):
			if( filename.is_file()):
				if (filename.name.endswith('.jpg') or filename.name.endswith('.jpeg') ):
					jpg += 1
				elif( filename.name.endswith('.png') ):
					png += 1
				else:					
					print("Ignoring file: ",filename.name)
					continue
				filepath = os.path.join( lpath, filename.name)
				image = cv2.imread(filepath,0)		# now read with cv2; the 0 indicates that it is loaded as greyscale
				#print("*",np.shape(image))
				image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
				#print("***",np.shape(image))
				sh = np.shape(image)	
				if(sh in dic):
					dic[sh] += 1
				else:
					dic[sh] = 1					
				#
				#save the new image
				cv2.imwrite( os.path.join( pathD, filename.name) , transform(image= image )["image"] )				
				#
				#images.append(image)
				cant += 1
				b = "Reaing-Saving images..." + str(cant)
				print (b, end="\r")		
	#		
	print("Total of images: ", cant)
	#
	print("jpg: ",jpg)
	print("png: ",png)
	return dic
				
	

def preproc_images(path,pathD,pixels=300):
	# preproc_images("Huble/setBGR/nebula","Huble/setBGR/8")
	# one path at once
	#tf.keras.utils.set_random_seed(0)	#for reproducible results	
	images = []
	#
	print("Reading images from: ",path)
	#
	cant = 0
	lpath = path	#os.path.join(imgpath, str(i))
	with os.scandir( lpath ) as ficheros:
		#print("len ficheros: ", len(ficheros))
		for filename in ficheros:
			#ficheros = [fichero.name for fichero in ficheros if fichero.is_file() and fichero.name.endswith('.jpg')]
			#if( filename.is_file() ):
			#print("fn: ",filename)
			#if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename.name):
			if( filename.is_file() and (filename.name.endswith('.jpg') or filename.name.endswith('.png')) ):
				cant=cant+1
				filepath = os.path.join( lpath, filename.name)
				image = cv2.imread(filepath,0)		# now read with cv2; the 0 indicates that it is loaded as greyscale
				print("*",np.shape(image))
				image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
				print("***",np.shape(image))
				#
				#scale and chop the image
				he = image.shape[0]	#alto
				wi = image.shape[1]	#ancho
				#
				if(wi<he):	
					# scale with respecto to width 
					image = imutils.resize(image, width=pixels)		# imutils to keep proportion
					# keep the center of the image
					center = image.shape[0]//2	# height of image
					p2 = pixels//2
					# crop image					
					image = image[center-p2:center+(pixels-p2),:]
				else:
					# scale with respect to heigth
					image = imutils.resize(image, height=pixels)		# imutils to keep proportion
					# keep the center of the image
					center = image.shape[1]//2	# height of image
					p2 = pixels//2
					# crop image					
					image = image[:,center-p2:center+(pixels-p2)]
				#save the new image
				cv2.imwrite( os.path.join( pathD, filename.name) ,image)				
				#
				#images.append(image)
				b = "Reaing-Saving images..." + str(cant)
				print (b, end="\r")
				#if prevRoot !=root:
				#	print(root, cant)
				#	prevRoot=root
				#	directories.append(root)
				#	dircount.append(cant)
				#	#cant=0		
	print("Total of images: ", cant)


def pp_image(path,pathD,name,pos="center"):
	# pp_image("Huble/setgray/Galaxia_Eliptica","Huble/setgray/1","eli6.jpg",pos="left")
	lpath = path	#os.path.join(imgpath, str(i))
	#
	filepath = os.path.join( lpath, name)
	image = cv2.imread(filepath,0)		# now read with cv2; the 0 indicates that it is loaded as greyscale
	print("*",np.shape(image))
	image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)		# this is done to recovery the 3 dimensions
	print("***",np.shape(image))
	#
	#scale and chop the image
	he = image.shape[0]	#alto
	wi = image.shape[1]	#ancho
	#
	if(wi<he):	
		# scale with respecto to width 
		image = imutils.resize(image, width=300)		# imutils to keep proportion
		# keep the center of the image
		center = image.shape[0]//2	# height of image
		# crop image					
		image = image[center-150:center+150,:]
	else:
		# scale with respect to heigth
		image = imutils.resize(image, height=300)		# imutils to keep proportion
		if(pos=="right"):
			image = image[:,-300:]
		elif(pos=="left"):
			image = image[:,:300]
		else:	# any other center
			# keep the center of the image
			center = image.shape[1]//2	# height of image
			# crop image					
			image = image[:,center-150:center+150]
	#save the new image
	cv2.imwrite( os.path.join( pathD, name) ,image)				
	#
	#images.append(image)
	#if prevRoot !=root:
	#	print(root, cant)
	#	prevRoot=root
	#	directories.append(root)
	#	dircount.append(cant)
	#	#cant=0		


def get_Galaxies_ext_8(imgpath = 'Galaxias_Totales_ext_8/'):	# 8 leaf nodes
	tf.keras.utils.set_random_seed(0)	#for reproducible results
	#imgpath = 'Galaxias_Totales_ext_8/' 
	 
	images = []
	directories = []
	dircount = []
	prevRoot=''
	cant=0

	print("leyendo imagenes de ",imgpath)
	for i in range(1,9):	# 8 dirs
		cant = 0
		lpath = os.path.join(imgpath, str(i))
		with os.scandir( lpath ) as ficheros:
			#print("len ficheros: ", len(ficheros))
			for filename in ficheros:
				#ficheros = [fichero.name for fichero in ficheros if fichero.is_file() and fichero.name.endswith('.jpg')]
				#if( filename.is_file() ):
				#print("fn: ",filename)
				#if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename.name):
				if( filename.is_file() and ( filename.name.endswith('.jpg') or filename.name.endswith('.png') ) ):
					cant=cant+1
					filepath = os.path.join( lpath, filename.name)
					image = plt2.imread(filepath)
					#image = plt.imread(filename)
					#image = Image.open(filepath)
					#print("**",np.shape(image))
					if((image.shape[0]!=300) or (image.shape[1]!=300)):
						print("*******imge: ",filename.name," : ",np.shape(image))
					images.append(image)
					#b = "Leyendo..." + str(cant)
					#print (b, end="\r")
					#if prevRoot !=root:
					#	print(root, cant)
					#	prevRoot=root
					#	directories.append(root)
					#	dircount.append(cant)
					#	#cant=0		
		print(i, cant)
		directories.append(str(i))
		dircount.append(cant)

	print('Directorios leidos:',len(directories))
	print("Imagenes en cada directorio", dircount)
	print('suma Total de imagenes en subdirs:',sum(dircount))

	labels = []
	c = 0
	for x in dircount:
		labels = np.concatenate([labels, [c for i in range(x)] ])
		c += 1

	#****

	print("Cantidad etiquetas creadas: ",len(labels))

	galaxies = directories.copy()	# change for real names
	print(galaxies)

	y = labels
	x = np.array(images, dtype="double")#it was "float32"	#/255.0 #convierto de lista a numpy
	#x = np.array(x[:,:,:,0], dtype=np.uint8) # requiere upper line	# but only one of rgb, because gray

	# Find the unique numbers from the train labels
	classes = np.unique(y)
	nClasses = len(classes)
	print('Total number of outputs : ', nClasses)
	print('Output classes : ', classes)


	#Mezclar todo y crear los grupos de entrenamiento y testing
	#train_X,test_X,train_Y,test_Y = train_test_split(x,y,test_size=0.2,random_state=0)
	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
	sss.get_n_splits(x, y)

	for i, (train_index, test_index) in enumerate(sss.split(x, y)):
		train_X = x[train_index]		
		train_Y = y[train_index]
		test_X = x[test_index]		
		test_Y = y[test_index]

	print('Training data shape : ', train_X.shape, train_Y.shape)
	print('Testing data shape : ', test_X.shape, test_Y.shape)


	# Change the labels from categorical to one-hot encoding
	train_Y_one_hot = to_categorical(train_Y)
	#test_Y_one_hot = to_categorical(test_Y)
	test_label  = to_categorical(test_Y)

	# Display the change for category label using one-hot encoding
	print('Original label:', train_Y[0])
	print('After conversion to one-hot:', train_Y_one_hot[0])


	#train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
	sss.get_n_splits(train_X, train_Y_one_hot)

	for i, (train_index, test_index) in enumerate(sss.split(train_X, train_Y_one_hot)):
		train_X_aux = train_X[train_index]		
		train_label = train_Y_one_hot[train_index]
		valid_X = train_X[test_index]		
		valid_label = train_Y_one_hot[test_index]

	train_X = train_X_aux
	 
	print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
	return (train_X,valid_X,train_label,valid_label,test_X,test_label)


def get_Galaxies_HC_ext_8(imgpath = 'Galaxias_Totales_ext_8/'):
	nameNodes = ["elliptical","S0","Sa","Sb","Sc","Sd","I","others","regular","spiral"]		#regular and spiral were aaded to get the hierarchy

	train_X,valid_X,train_label,valid_label,test_X,test_label = get_Galaxies_ext_8(imgpath)	

	#load hierarchy 
	ld = ldata()
	structure = ld.loadHierachyArff("galaxies_ext_8.hierarchy",nameNodes)
	H = hierarchy(structure,nameNodes)
	H.initialize()

	# process the labels to be consisntent with the hierarchy
	shTY = np.shape(train_label)
	ntrain_label = np.zeros( (shTY[0], len(nameNodes)),dtype=bool)
	ntrain_label[:,:shTY[1]] = train_label[:,:]
	train_label = ntrain_label
	H.forceConsistency(train_label,overwrite=True)

	shTY = np.shape(test_label)
	ntest_label = np.zeros( (shTY[0], len(nameNodes)),dtype=bool)
	ntest_label[:,:shTY[1]] = test_label[:,:]
	test_label = ntest_label
	H.forceConsistency(test_label,overwrite=True)

	shTY = np.shape(valid_label)
	nvalid_label = np.zeros( (shTY[0], len(nameNodes)),dtype=bool)
	nvalid_label[:,:shTY[1]] = valid_label[:,:]
	valid_label = nvalid_label
	H.forceConsistency(valid_label,overwrite=True)
	
	return (train_X,valid_X,train_label,valid_label,test_X,test_label,H)


def  imageAugmentation(trainSet,cl,H,minI=None,pathSave=None):
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
	tf.keras.utils.set_random_seed(0)

	#define the pipeline with the tranformations
	transform = A.Compose([
		A.HorizontalFlip(always_apply=False, p=0.5),
		A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.2, 0.2), rotate_limit=(-20, 20), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
		A.Resize(always_apply=False, p=1.0, height=300, width=300, interpolation=0)
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

