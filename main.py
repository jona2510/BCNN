

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from hierarchicalClassification.BNHC_C_CNN import BCNN as hu
from load_galaxiesDS import get_Galaxies_HC_ext_8
import hierarchicalClassification.evaluationHC as ehc
from hierarchicalClassification.SPL_PP import TopDownProcedure as TD
import numpy as np


# LOAD GALAXIES IMAGES
train_X,valid_X,train_label,valid_label,test_X,test_label,H = get_Galaxies_HC_ext_8(imgpath = 'Galaxies_HC_DS/')	
 
# BCNN without image augmentation:
hc = hu(hierarchy=H, smooth=0.1, BATCH_SIZE=64, epochs=30, nameNet="galaxies_8_cont_cw.net" )
#uncomment to carry out image augmentation:
#hc = hu(hierarchy=H, smooth=0.1, BATCH_SIZE=8, epochs=30, weighted=False, search_thr=False, nameNet="galaxies_8_cont_cw.net",augmentation=True, pathAugmentation="AD_BN+CNN" )	# 

# change to True to carry out fine_tunning (CNN parameter):
hc.do_fine_tuning = False	

# fit the model
hc.fit(train_X,train_label, valid_X,valid_label)
ni = len(test_X)

# predictions
test_pred = hc.predict_proba(test_X[:ni])	
# Top Down
test_pred2 = TD(test_pred,H)

# evaluation
print("BCNN scores (exact match, accuracy, hR, hP, hF)\n", ehc.exactMatch(test_label,test_pred2), ehc.hAccuracy(test_label,test_pred2), ehc.hRecall(test_label,test_pred2) , ehc.hPrecision(test_label,test_pred2) , ehc.hFmeasure(test_label,test_pred2) )






