from Models import *
from DataLoader import MyDataset

# groups = {'group1':[79, 88, 75, 42, 51, 58, 93, 1, 59, 55, 53, 76],
# 	      'group2':[17, 46, 63, 90, 16, 48, 87, 30], # sbr, cbr
# 	      'group3':[23, 77, 21, 6, 43, 39, 83, 67, 31, 5, 24, 56], # ser
# 	      'group4':[91, 69, 12, 80, 19, 26, 70, 28, 61, 47, 65, 0, 13, 14, 18], # brsh, brlo, brge, brlt
# 	      'group5':[9, 10, 44, 45, 49, 60, 64, 78, 84, 85, 7, 25, 38, 41, 50, 54, 72, 81, 86, 92],
# 	      'group6':[89, 62, 52, 36, 4, 3, 73, 74, 11, 33, 15, 82, 94, 66],
# 	      'group7':[40, 57, 2, 8, 68, 71],
# 	      'group8':[37, 95, 22],
# 	      'group9':[20, 27, 29, 32, 34, 35]}

groups_lengths = ['group1':12, 'group2':8, 'group3':12, 'group4':15, 'group5':20, 'group6':14, 'group7':6, 'group8':3, 'group9':6, 'All':96]

def train(data_path,
		  labels_path,
		  group,
		  number_of_datafiles,
		  stack_number,
		  model,
		  num_epochs=50):
	
	my_data = MyDataset(data_path=data_path,
						labels_path=labels_path,
						group=group,
						number_of_datafiles=number_of_datafiles,
						stack_number=stack_number,
						transforms=None)

	train_size = len(my_data) - (0.1*(len(my_data)))
	test_size = 0.1*(len(my_data))

	# Random split
	train_dataset, test_dataset = torch.utils.data.random_split(my_data, [train_size, test_size])

	# Dataloaders
	batch_size = 8

	train_loader = DataLoader(train_dataset, drop_last=True, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, drop_last=True, batch_size=batch_size, shuffle=True)

	# Load model
	if model == 'ResNet':
		layers=[3, 4, 6, 3]
		model = MyResNet(BasicBlock, layers, num_classes=len(groups_lengths[group]))
	elif model == 'Vgg':
		model = Vgg(num_classes=len(groups_lengths[group]))
	else:
		raise Exception("The selected model is not correct!")
	
	if torch.cuda.is_available():
	    model.cuda()
	    print('cuda is ok')

	# Loss function
	criterion = nn.CrossEntropyLoss()

	# Optimizer
	learning_rate = 0.0001

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

	# Pre-requests initialization
	train_loss = []
	test_loss = []

	train_accuracy = []
	test_accuracy = []

	dim = groups_lengths[group]
	train_confusion_matrix = torch.zeros(dim, dim, dtype=int)
	test_confusion_matrix = torch.zeros(dim, dim, dtype=int)

	#########################################################################################
	# training
	# shape_0 = 50
	# shape_1 = 51*5

	iter = 0
	for epoch in range(num_epochs):
	    t_l = 0
	    train_total = 0
	    train_correct = 0
	    for i, (images, train_labels) in enumerate(train_loader):
	        if torch.cuda.is_available():
	            images = Variable(images.view(batch_size, stack_number, -1).float().cuda())
	            train_labels = Variable(train_labels.view(batch_size).float().cuda())
	        else:
	            images = Variable(images.view(batch_size, stack_number, -1))
	            train_labels = Variable(train_labels.view(batch_size))

	        optimizer.zero_grad()
	        
	        train_outputs = model(images)

	        loss = criterion(train_outputs, train_labels.long())
	        t_l += loss.data
	        
	        loss.backward()
	        
	        optimizer.step()
	        
	        iter += 1
	        
	        train_total += train_labels.size(0)
	        _, predicted = torch.max(train_outputs.data, 1)
	        train_correct += (predicted.cpu() == train_labels.data.cpu()).sum()
	        
	        for t, p in zip(train_labels.data.view(-1), predicted.view(-1)):
	            train_confusion_matrix[t.long(), p.long()] += 1
	                
	        if iter % len(train_loader) == 0:
	            correct = 0
	            total = 0
	            s_l = 0
	            with torch.no_grad():
	                for j, (images, test_labels) in enumerate(test_loader):
	                    if torch.cuda.is_available():
	                        images = Variable(images.view(batch_size, stack_number, -1).float().cuda())
	                        test_labels = Variable(test_labels.view(batch_size).float().cuda())
	                    else:
	                        images = Variable(images.view(batch_size, stack_number, -1))
	                        test_labels = Variable(test_labels.view(batch_size))
	                    
	                    test_outputs = model(images.float())

	                    _, predicted = torch.max(test_outputs.data, 1)

	                    loss = criterion(test_outputs, test_labels.long())
	                    s_l += loss

	                    total += test_labels.size(0)
	                    
	                    correct += (predicted.cpu() == test_labels.data.cpu()).sum()
	                    
	                    for t, p in zip(test_labels.data.view(-1), predicted.view(-1)):
	                        test_confusion_matrix[t.long(), p.long()] += 1
	                    
	                    del test_outputs, images
	            
	#             PATH = save_path + 'models/model' + str(epoch+1) + '.pth'
	#             torch.save(model, PATH)
	            accuracy = 100 * correct / total
	            train_acc = 100 * train_correct / train_total
	            
	            t_l = t_l/len(train_loader)
	            s_l = s_l/len(test_loader)
	            
	            train_loss.append(t_l.cpu())
	            test_loss.append(s_l.cpu())
	            
	            train_accuracy.append(train_acc.cpu())
	            test_accuracy.append(accuracy.cpu())
	            
	#             print('Epoch: {}. Train Loss: {}. Test Loss: {}. Train_ACC: {}. ACC: {}'.format(epoch+1, t_l, s_l, train_acc, accuracy))
	            print('Epoch: {0:1}. Train Loss: {1:2.4f}. Test Loss: {2:2.4f}. Train_ACC: {3:2.2f}. ACC: {4:2.2f}'.format(epoch+1, t_l, s_l, train_acc, accuracy))

# if __name__ == '__main__':
# 	train()
