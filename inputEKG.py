''' pipeline for feeding data into the network'''

import scipy.io as sc
import numpy as np
global data
data = sc.loadmat('DATA_TF/dataEKG_training.mat')
data =data['dataEKG_training']

global labels
labels = sc.loadmat('DATA_TF/labelsEKG_training.mat')
labels =labels['labelsEKG_training']
#labels = labels[:,:300]
print(labels.shape)
labels = np.reshape(labels.T, (300,))
labels = labels -1
data = np.transpose(data, (3,0,1,2))

#data = data[:300,:,:,:]

weights = sc.loadmat('weights.mat')
weights_ft=weights['weights_c']


weights_ft[0][3]= np.squeeze(weights_ft[0][3], axis=(0,1))
weights_ft[0][4]= np.squeeze(weights_ft[0][4], axis=(0,1))
