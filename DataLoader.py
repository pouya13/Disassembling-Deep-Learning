import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, labels_path, number_of_datafiles=2 transforms=None):
    	self.data_path = data_path
    	self.labels_path = labels_path
    	
        self.data_files = os.listdir(data_path)
        self.labels_path = os.listdir(labels_path)
        self.number_of_datafiles = number_of_datafiles
        self.transforms = transforms

        if len(self.data_files) == 0:
        	raise Exception("The data directory is empty!")

        if len(self.labels_files) == 0:
        	raise Exception("The labels directory is empty!")

        if number_of_datafiles < 2:
        	raise Exception("The number of data files should be greater than one!")

	
	def load_data_files(self):
		train_data1 = np.loadtxt(path + 'data1.txt', delimiter=',')
		train_labels1 = np.loadtxt(path + 'labels1.txt', delimiter=',')

		train_data2 = np.loadtxt(path + 'data2.txt', delimiter=',')
		train_labels2 = np.loadtxt(path + 'labels2.txt', delimiter=',')

		train_data = np.vstack((train_data1, train_data2))
		del train_data1, train_data2

		for i in range(2, min(self.number_of_datafiles, len(self.data_files))):
			train_data1 = np.loadtxt(path + 'data1.txt', delimiter=',')
			train_labels1 = np.loadtxt(path + 'labels1.txt', delimiter=',')

			train_data2 = np.loadtxt(path + 'data2.txt', delimiter=',')
			train_labels2 = np.loadtxt(path + 'labels2.txt', delimiter=',')

			train_data = np.vstack((train_data1, train_data2))
			del train_data1, train_data2
			print(np.shape(train_data))
         
    def __len__(self):
        return (len(self.x))
    
    def __getitem__(self, i):
        data = self.x[i]
        data = np.asarray(data, dtype=np.float64).reshape(50,51*5)
        
        if self.transforms:
            data = self.transforms(data)
        
#         y = np.zeros((self.dim, 1), dtype=np.float64)
#         y = self.y[i] * np.ones((5))
            
        return (data, self.y[i])
