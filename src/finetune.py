import argparse
import torch
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from ContrastiveLossModel import ContrastiveLossModel

class KinectDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions, movement, label = self.data[idx]
        # Convert data to PyTorch tensors and label to float32
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32) # 0 or 1
        return joint_positions, label

    def filter_by_exercise(self, exercise):
        # Filter the data based on the movement type/number
        filtered_data = [data for data in self.data if data[1] == exercise]
        return KinectDataset(filtered_data) 

    def get_subset(self, indices):
        return KinectDataset([self.data[i] for i in indices])

    def append(self, new_data):
        self.data.extend(new_data)

class LinearParadigm(nn.Module):
	def __init__(self, f, A):
		super(LinearParadigm, self).__init__()

		self.f_q = f
		self.A = A

		for param in self.f_q.parameters():
			param.requires_grad = False

		self.mlp = nn.Sequential(
			nn.Linear(256,256), 
			nn.ReLU(), 
			nn.Linear(256,1))
	def forward(self, x):
		return self.mlp(self.f_q(x, self.A))

def get_dataset(dataset):
	if dataset == 'prmd':
		pickle_path = 'data/clean/uiprmd.pkl'
	elif dataset == 'ir':
		pickle_path = 'data/clean/intellirehab.pkl'

	with open(pickle_path,'rb') as f:
		data = pickle.load(f)

	cleanData = KinectDataset(data)

	all_lengths = [len(sample[0]) for sample in cleanData]

	avg_length = int(np.mean(all_lengths))
	# avg_length = 400

	def resample_positions(dataset):
		for sample in dataset.data:
			joint_positions = sample[0]
			diff = avg_length - len(joint_positions)

			if diff < 0:
				indices = np.linspace(0, len(joint_positions) - 1, avg_length).astype(int)
				joint_positions = joint_positions[indices]
			elif diff > 0:
				last_position = joint_positions[-1]
				joint_positions = np.vstack([joint_positions, np.tile(last_position, (diff, 1, 1))])

			sample[0] = joint_positions

	resample_positions(cleanData)
	return cleanData

def get_dataloader(clean_data, exercise, bs = 1):
	torch.manual_seed(42)
	train_len = int(0.8 * len(clean_data))
	val_len = len(clean_data) - train_len

	train_data, val_data = random_split(clean_data, [train_len,val_len])

	train_exercise = clean_data.get_subset(train_data.indices).filter_by_exercise(exercise)
	val_exercise = clean_data.get_subset(val_data.indices).filter_by_exercise(exercise)

	train_loader = DataLoader(train_exercise, batch_size=bs, shuffle=True)
	val_loader = DataLoader(val_exercise, batch_size=1, shuffle=False)
	return train_loader, val_loader

def create_adj(node_dim):
	if node_dim == 25:
		connectivity = [(0, 1), (0, 12) , (0, 16) , (1, 20) , (12, 13) , (13, 14) , (14, 15) , (16, 17) , (17, 18) , (18, 19) , (20, 4) , (20, 8) , (20, 2) , (2, 3) , (4, 5) , (5, 6) , (6, 7) , (6, 22) , (7, 21) , (8, 9) , (9, 10) , (10, 11) , (10, 24) , (11, 23)]
	elif node_dim == 22:
		connectivity = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0), (3, 6), (6, 7), (7, 8), (8, 9),  (3, 10), (10, 11), (11, 12), (12, 13),  (0, 14), (14, 15), (15, 16), (16, 17), (0, 18), (18, 19), (19, 20), (20, 21)]

# Create source and target node lists
	src_nodes = [i for i, j in connectivity]
	tgt_nodes = [j for i, j in connectivity]

# Include reversed connections to make the edges undirected
	src_nodes += tgt_nodes
	tgt_nodes += [i for i, j in connectivity]

# Convert lists to tensors
	adj = torch.tensor([src_nodes, tgt_nodes])

	return adj

def main():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--modelName', type=str, help='ModelName')
	parser.add_argument('--dataset', type=str, help='Dataset')
	parser.add_argument('--exercise', type=str, help='Exercise')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../best_models/PRMD/'))

	if args.dataset == 'prmd':
		node_dim = 22
	elif args.dataset == 'ir':
		node_dim = 25

	adj = create_adj(node_dim)
	model = ContrastiveLossModel(3, node_dim, adj, device).to(device)

	model_path = os.path.join(models_directory, args.modelName)
	model.load_state_dict(torch.load(model_path, map_location=device))

	newModel = LinearParadigm(model.f_q, model.A).to(device)

	clean_data = get_dataset(args.dataset)
	train_loader, val_loader = get_dataloader(clean_data, args.exercise, 64)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(newModel.parameters(), lr = 0.0001)
	num_epochs = 2000

	for epoch in range(num_epochs):
		train_loss, train_acc = train(newModel,train_loader,criterion,optimizer,device)
		val_loss, val_acc = validate(newModel,val_loader,criterion,device)

		print("Epoch: ", epoch, " - Training Loss: ", train_loss, " - Validation Loss: ", val_loss, " - Training Acc: ", train_acc, " - Validation Acc: ", val_acc, flush = True)	

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    training_loss = 0.0
    correct = 0
    total = 0
    for joint_positions, labels in train_loader:
        joint_positions = joint_positions.to(device)
        labels = labels.to(device)
        outputs = model(joint_positions).squeeze(1)
        loss = criterion(outputs, labels)

        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * joint_positions.size(0)
    return training_loss / len(train_loader.dataset), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    preds = []
    _labels = []
    validation_loss = 0.0
    correct = 0
    total = 0
    for joint_positions, labels in val_loader:
        joint_positions = joint_positions.to(device)
        labels = labels.to(device)
        outputs = model(joint_positions).squeeze(1)
        loss = criterion(outputs, labels)

        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        validation_loss += loss.item() * joint_positions.size(0)
        _labels.append(labels.item())
        preds.append(torch.sigmoid(outputs).item())
    global val_max
    if correct / total >= val_max:
        val_max = correct / total
        print(preds, flush = True)
        print(_labels, flush = True)
    return validation_loss / len(val_loader.dataset), correct / total

val_max = float('-inf')
if __name__ == '__main__':
	main()
