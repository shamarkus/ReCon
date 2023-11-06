import torch
import gc
import re
import pickle
import numpy as np
import os
import pickle
import argparse
import torch.nn.functional as F
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
	train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
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
	parser.add_argument('--modelName', type=str, help='Model Name')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))
	model_path = os.path.join(models_directory, args.modelName)

	clean_data = get_dataset(args.dataset)
	train_loader, val_loader = get_dataloader(clean_data, 10)
		
	if args.dataset == 'prmd':
		node_dim = 22
	elif args.dataset == 'ir':
		node_dim = 25

	adj = create_adj(node_dim)

	model = ContrastiveLossModel(3, node_dim, adj, device).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))

	get_masters(model, train_loader)
	train_preds, train_truths, train_grads = get_norm(model, train_loader)
	val_preds, val_truths, val_grads = get_norm(model, val_loader)

	with open(model_path + '_preds.pkl', 'wb') as file:
		pickle.dump({
		'train_preds': train_preds, 
		'train_truths': train_truths, 
		'train_grads': train_grads,
		'val_preds': val_preds, 
		'val_truths': val_truths,
		'val_grads': val_grads,
		'master_embs': model.cor_emb.detach().numpy(),
		}, file)


def get_masters(model, train_loader):
	model.eval()
	model.cor_emb = torch.zeros(10,128).to(model.device)

	for joint_positions, labels in train_loader:
		indices = torch.where(labels >= 0)[0]
		joint_positions = joint_positions[indices]
		labels = labels[indices].long()

		model.get_norm(joint_positions, labels)
	
	model.apply_avg()

def get_norm(model, train_loader):
	model.eval()

	adj = torch.sparse_coo_tensor(model.A, torch.ones(model.A.shape[-1]).to(model.device)).to_dense().to(model.device).requires_grad_(False)
	adj = adj + torch.eye(adj.shape[0]).to(model.device)

	preds = []
	truths = []
	cams = []

	for joint_positions, labels in train_loader:
		labels = labels.to(model.device).long()

		joint_positions = joint_positions.to(model.device)
		h, _ = model.f_q.ste((model.f_q.ri(joint_positions), model.A))
		z = model.g_q(model.f_q.pool(h))

		preds.extend(z.cpu().detach().numpy())
		truths.extend(labels.cpu().detach().numpy())

		cams.extend(get_cams(model, adj, h, z, labels))

	return (np.array(preds), np.array(truths), np.array(cams))	

def get_cams(model, adj, h, z, labels):
	B, T, N, C = h.shape
	labels = torch.where(labels < 0, -1 * (labels + 1), labels)
	masters = model.cor_emb[labels]

	cosine_sim = (masters * z).sum(dim = 1)

	h.requires_grad_(True)
	h.retain_grad()
	cosine_sim.backward(torch.ones_like(cosine_sim))

	grads = h.grad # Something along the lines of BxTxNxC

	imp = torch.mean(F.relu(grads), dim = [1,2])
	feats = F.relu((h * imp.view(B, 1, 1, C)).sum(dim = 3))
	return torch.einsum('btn,nj->btj', feats, adj).cpu().detach().numpy()

if __name__ == '__main__':
	main()
