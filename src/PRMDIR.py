import torch
import gc
import re
import pickle
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from ContrastiveLossModel import ContrastiveLossModel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, accuracy_score

class KinectDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions, movement, label = self.data[idx]
        # Convert data to PyTorch tensors and label to float32
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        movement_indx = (int(re.findall(r'\d+',movement)[0]) % 10)
        label = torch.tensor((-1 * (movement_indx + 1)) if (label == 0) else movement_indx, dtype=torch.float32)
        return joint_positions, label

    def get_labels(self):
        return torch.tensor([self[i][1] for i in range(len(self.data))], dtype=torch.float32)

    def filter_by_exercise(self, exercise):
        # Filter the data based on the movement type/number
        filtered_data = [data for data in self.data if data[1] == exercise]
        return KinectDataset(filtered_data) 

    def get_subset(self, indices):
        return KinectDataset([self.data[i] for i in indices])

    def append(self, new_data):
        self.data.extend(new_data)

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
	
def get_dataloader(clean_data, bs = 1):
	torch.manual_seed(42)
	train_len = int(0.8 * len(clean_data))
	val_len = len(clean_data) - train_len

	train_data, val_data = random_split(clean_data, [train_len,val_len])
	train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
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
	parser.add_argument('--dataset', type=str, help='Dataset')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))

	clean_data = get_dataset(args.dataset)
	train_loader, val_loader = get_dataloader(clean_data, 128)
		
	if args.dataset == 'prmd':
		node_dim = 22
	elif args.dataset == 'ir':
		node_dim = 25

	adj = create_adj(node_dim)
	model = ContrastiveLossModel(3, node_dim, adj, device).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

	num_epochs = 2000

	for epoch in range(num_epochs):
		train_loss = train(model, train_loader, optimizer)
		get_norm(model, train_loader)
		validate(model, val_loader)

		if epoch % 15 == 0:
			# torch.save(model.state_dict(), os.path.join(models_directory,f"PRMD_E{epoch+1}_T{train_loss:.4f}_V{val_loss:.4f}.pth"))
			torch.save(model.state_dict(), os.path.join(models_directory,f"PRMD_E{epoch+1}_T{train_loss:.4f}.pth"))

		# print("Epoch: ", epoch, " - Training Loss: ", train_loss, " - Validation Loss: ", val_loss, flush = True)
		print("Epoch: ", epoch, " - Training Loss: ", train_loss, flush = True)

def train(model, train_loader, optimizer):
	model.train()
	training_loss = 0.0
	for joint_positions, labels in train_loader:
		loss = model(joint_positions, labels)
		training_loss += loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return training_loss / len(train_loader)

def get_norm(model, train_loader):
	model.eval()
	model.cor_emb = torch.zeros(10,128).to(model.device)

	# cor_count = torch.zeros(10)

	for joint_positions, labels in train_loader:
		indices = torch.where(labels >= 0)[0]
		joint_positions = joint_positions[indices]
		labels = labels[indices].long()

		# cor_count += torch.bincount(labels, minlength = 10)
		model.get_norm(joint_positions, labels)
	
	# model.apply_avg(cor_count)
	model.apply_avg()

def validate(model, val_loader):
	model.eval()
	preds = [[] for _ in range(10)]
	cors = [[] for _ in range(10)]

	for joint_positions, labels in val_loader:
		cor_indices = torch.where(labels >= 0)[0]
		inc_indices = torch.where(labels < 0)[0]

		cor_pos = joint_positions[cor_indices]
		cor_labels = labels[cor_indices].long()

		inc_pos = joint_positions[inc_indices]
		inc_labels = (-1 * (labels[inc_indices] + 1)).long()

		cor_preds = model.validate(cor_pos, cor_labels)
		inc_preds = model.validate(inc_pos, inc_labels)

		[preds[i.item()].extend(cor_preds[cor_labels == i].tolist()) for i in cor_labels.unique()]
		[cors[i.item()].extend((cor_labels == i).sum() * [1]) for i in cor_labels.unique()]

		[preds[i.item()].extend(inc_preds[inc_labels == i].tolist()) for i in inc_labels.unique()]
		[cors[i.item()].extend((inc_labels == i).sum() * [0]) for i in inc_labels.unique()]

	for i in range(10):
		if len(preds[i]):
			cosine_sim = np.array(preds[i])
			labels = np.array(cors[i])

			roc_auc = roc_auc_score(labels, cosine_sim)

			pr_auc = average_precision_score(labels, cosine_sim)

			fpr, tpr, thresholds = roc_curve(labels, cosine_sim)

			dists = np.sqrt((1 - tpr)**2 + fpr**2)
			best_threshold = thresholds[np.argmin(dists)]

			binary_preds = (cosine_sim >= best_threshold).astype(int)
			accuracy = accuracy_score(labels, binary_preds)

			print(f"Exercise: {i}, ROC_AUC: {roc_auc:.4f}, PR_AUC: {pr_auc:.4f}, THRSH: {best_threshold:.4f}, ACC: {accuracy:.4f}", flush = True)

if __name__ == '__main__':
	main()
