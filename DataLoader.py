import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
	def __init__(self, data_path, labels_path, group, number_of_datafiles=2, stack_number=50, transforms=None):
		self.group = group
		self.data_path = data_path
		self.labels_path = labels_path
		self.stack_number = stack_number

		self.data_files = os.listdir(data_path)
		self.label_files = os.listdir(labels_path)
		self.number_of_datafiles = number_of_datafiles
		self.transforms = transforms

		self.groups = {'group1':[79, 88, 75, 42, 51, 58, 93, 1, 59, 55, 53, 76],
					   'group2':[17, 46, 63, 90, 16, 48, 87, 30], # sbr, cbr
					   'group3':[23, 77, 21, 6, 43, 39, 83, 67, 31, 5, 24, 56], # ser
					   'group4':[91, 69, 12, 80, 19, 26, 70, 28, 61, 47, 65, 0, 13, 14, 18], # brsh, brlo, brge, brlt
					   'group5':[9, 10, 44, 45, 49, 60, 64, 78, 84, 85, 7, 25, 38, 41, 50, 54, 72, 81, 86, 92],
					   'group6':[89, 62, 52, 36, 4, 3, 73, 74, 11, 33, 15, 82, 94, 66],
					   'group7':[40, 57, 2, 8, 68, 71],
					   'group8':[37, 95, 22],
					   'group9':[20, 27, 29, 32, 34, 35]}

		if len(self.data_files) == 0:
			raise Exception("The data directory is empty!")

		if len(self.labels_files) == 0:
			raise Exception("The labels directory is empty!")

		if number_of_datafiles < 2:
			raise Exception("The number of data files should be greater than one!")

		self.load_data_files()

	
	def load_data_files(self):
		print("Loading data and labels!")
		data1 = np.loadtxt(os.path.join(self.data_path, self.data_files[0]), delimiter=',')
		labels1 = np.loadtxt(os.path.join(self.data_path, self.label_files[0]), delimiter=',')

		data2 = np.loadtxt(os.path.join(self.data_path, self.data_files[1]), delimiter=',')
		labels2 = np.loadtxt(os.path.join(self.data_path, self.label_files[1]), delimiter=',')

		data = np.vstack((data1, data2))
		labels = np.hstack((labels1, labels2))
		del data1, data2, labels1, labels2

		for i in range(2, min(self.number_of_datafiles, len(self.data_files))):
			data_temp = np.loadtxt(os.path.join(self.data_path, self.data_files[i]), delimiter=',')
			labels_temp = np.loadtxt(os.path.join(self.data_path, self.label_files[i]), delimiter=',')

			data = np.vstack((data, data_temp))
			labels = np.hstack((labels, labels_temp))
			del data_temp, labels_temp

		self.group_selection(data, labels)

	def group_selection(self, data, labels):
		x = []
		y = []
		if self.group in list(self.groups.keys()):
			for i in range(len(data)):
				if labels[i] in self.groups[self.group]:
					x.append(data[i])
					y.append(self.groups[self.group].index(labels[i]))

			data = x
			labels = y

			del x, y

			self.pre_process(data,labels)

		elif self.group == 'All':
			self.pre_process(data, labels)
		else:
			raise Exception("Please, select a proper group!")

	def pre_process(self, data, labels):
		x = []
		y = []
		for j in range(int(max(labels)+1)):
			z = []
			for i in range(len(data)):
				if labels[i] == j:
					_, _, Zxx = signal.stft(data[i], 5e8, nperseg=100)
					z.append(np.abs(Zxx))
				if len(z) == self.stack_number:
					x.append(z)
					y.append(labels[i])
					del z
					z = []
		self.data = x
		self.labels = y

		del x, y, Zxx, data, labels

	def __len__(self):
	    return (len(self.x))

	def __getitem__(self, i):
		data = self.data[i]
		data = np.asarray(data, dtype=np.float64).reshape(self.stack_number,-1)

		if self.transforms:
			data = self.transforms(data)
		return (data, self.labels[i])
