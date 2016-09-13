''' pipeline for feeding data into the network'''

import scipy.io as sc
import numpy as np
global data
data = sc.loadmat('DATA_TF/dataEKG_training.mat')
data =data['data']
global labels
labels = sc.loadmat('DATA_TF/labelsEKG_training.mat')
labels =labels['labels']
labels = np.reshape(labels.T, (7657,))
labels = labels -1
data = np.transpose(data, (3,0,1,2))


