import argparse
import torch
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ContrastiveRegressiveModel import ContrastiveRegressiveModel
from KimoreDataset import createKimoreAdjacencyMatrix, KimoreCustomDataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.utils.data import DataLoader

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

def split_datasets(S_dataset, T_dataset):
    # movements = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']
    movements = ['Es1']

    S_train_data = []
    T_train_data = []
    S_val_data = []
    T_val_data = []

    for movement in movements:
# Split for S_dataset
        S_filtered = S_dataset.filter_by_exercise(movement)
        S_train_subset, S_val_subset = train_test_split(S_filtered.data, test_size=0.2, random_state=42)
        S_train_data.extend(S_train_subset)
        S_val_data.extend(S_val_subset)

# Split for T_dataset
        T_filtered = T_dataset.filter_by_exercise(movement)
        T_train_subset, T_val_subset = train_test_split(T_filtered.data, test_size=0.2, random_state=42)
        T_train_data.extend(T_train_subset)
        T_val_data.extend(T_val_subset)

# Convert back to KimoreCustomDataset
    S_train = KimoreCustomDataset(S_train_data)
    T_train = KimoreCustomDataset(T_train_data)
    S_val = KimoreCustomDataset(S_val_data)
    T_val = KimoreCustomDataset(T_val_data)

    return S_train, T_train, S_val, T_val

class LinearParadigm(nn.Module):
	def __init__(self, f, A, A_dense):
		super(LinearParadigm, self).__init__()

		self.f_q = f
		self.A = A
		self.A_dense = A_dense
		# self.g_q = g

		for param in self.f_q.parameters():
			param.requires_grad = False

		# for param in self.g_q.parameters():
		# 	param.requires_grad = False

		self.mlp = nn.Sequential(
			nn.Linear(256,256), 
			nn.ReLU(), 
			nn.Linear(256,1))
	def forward(self, x):
		# return self.mlp(self.g_q(self.f_q(x)))
		return self.mlp(self.f_q(x, self.A, self.A_dense))
			

def main():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--modelName', type=str, help='ModelName')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))

	adj = createKimoreAdjacencyMatrix()

	model = ContrastiveRegressiveModel(3,25, adj, device).to(device)

	model_path = os.path.join(models_directory, args.modelName)
	model.load_state_dict(torch.load(model_path, map_location=device))

	newModel = LinearParadigm(model.f_q, model.A, model.A_dense)
	newModel = newModel.to(device)

	S_dataset, T_dataset = get_datasets()

	S_train, T_train, S_val, T_val = split_datasets(S_dataset, T_dataset)

	training_data = KimoreCustomDataset(S_train.data + T_train.data)
	validation_data = KimoreCustomDataset(S_val.data + T_val.data)

	train_loader = DataLoader(training_data, batch_size=10, shuffle=True)
	val_loader = DataLoader(validation_data, batch_size=1, shuffle=True)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(newModel.parameters(), lr = 0.0008)

	num_epochs = 2000

	for epoch in range(num_epochs):
		train_loss = train(newModel,train_loader,criterion,optimizer,device)
		val_loss = validate(newModel,val_loader,criterion,device)

		print("Epoch: ", epoch, " - Training Loss: ", train_loss, " - Validation Loss: ", val_loss, flush = True)	

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    training_loss = 0.0
    for joint_positions, labels in train_loader:
        joint_positions = joint_positions.to(device)
        labels = labels.to(device)
        outputs = model(joint_positions)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * joint_positions.size(0)
    return training_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    validation_loss = 0.0
    for joint_positions, labels in val_loader:
        joint_positions = joint_positions.to(device)
        labels = labels.to(device)
        outputs = model(joint_positions)
        loss = criterion(outputs, labels)

        validation_loss += loss.item() * joint_positions.size(0)
    return validation_loss / len(val_loader.dataset)

if __name__ == '__main__':
	main()
