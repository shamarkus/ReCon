import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from STGCN import STEncoder
from ContrastiveRegressiveModel import ContrastiveRegressiveModel
from KimoreDataset import createKimoreAdjacencyMatrix, KimoreDataLoader, KimoreCustomDataset

def get_datasets():
	with open('data/clean/kimoreDataset.pkl','rb') as f:
		data = pickle.load(f)
	cleanData = KimoreCustomDataset(data)
	# Splitting into exemplar and non-exemplar
	S_dataset = cleanData.filter_by_exemplar_group()
	T_dataset = cleanData.filter_by_exemplar_group(False)
	# Calculate average temporal length
	all_lengths = [len(sample[0]) for sample in cleanData]
	# avg_length = int(np.mean(all_lengths))
	avg_length = 400
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
	resample_positions(S_dataset)
	resample_positions(T_dataset)
	return S_dataset, T_dataset

class Regressor(nn.Module):
	def __init__(self):
		super(Regressor, self).__init__()
		self.f = ContrastiveRegressiveModel(3,25, createKimoreAdjacencyMatrix(), torch.device('cuda'))
		self.fc1 = nn.Linear(256*25,1)
		# self.fc2 = nn.Linear(256,1)

	def forward(self, x):
		emb = self.f.f_q(x, self.f.A, self.f.A_dense)
		emb = emb.view(1,-1) # 1x6*128
		return self.fc1(emb)

def main():
# Create an instance of the custom DataLoader
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = Regressor().to(device)
	criterion = nn.MSELoss()

	S_dataset, T_dataset = get_datasets()
	clean_data = KimoreCustomDataset(S_dataset.data + T_dataset.data)

	subset_data = KimoreCustomDataset(clean_data.data[-8:])
	kimore_loader = DataLoader(subset_data, batch_size=1, shuffle=True)

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

	for i, layer in enumerate(model.f.f_q.ste):
		layer.register_backward_hook(print_grad_norm)

	model.fc1.register_backward_hook(print_grad_norm)
	# model.fc2.register_backward_hook(print_grad_norm)
		# Sample: How to use it in a training loop
	num_epochs = 600
	for epoch in range(num_epochs):
		for joint_positions, labels in kimore_loader:
			running_loss = 0.0
			# Your training code here
			# model, optimizer, loss_function, etc.
			joint_positions = joint_positions.to(device)
			labels = labels.to(device).unsqueeze(0)
			
			outputs = model(joint_positions)
			print(labels, outputs)
			loss = criterion(outputs, labels)
			
			running_loss += loss.item() * joint_positions.size(0)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print("Epoch ", epoch, flush = True)
		print("Loss ", running_loss / len(kimore_loader.dataset), flush = True)

		# Return the number of batches this DataLoader will produce in one epoch

# Function to print gradient norms
def print_grad_norm(module, grad_input, grad_output):
	if grad_input:
		grad_norm = grad_input[0].norm().item()
		print(f"{module.__class__.__name__} grad norm: {grad_norm}")

if __name__ == '__main__':
	main()
