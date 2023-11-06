import pickle
import numpy as np
import torch
import os

from ContrastiveRegressiveModel import ContrastiveRegressiveModel
from KimoreDataset import createKimoreAdjacencyMatrix, KimoreDataLoader, KimoreValLoader, KimoreCustomDataset 
from sklearn.model_selection import train_test_split

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
	movements = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']

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

	
def main():
# Create an instance of the custom DataLoader
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	adj = createKimoreAdjacencyMatrix()

	model = ContrastiveRegressiveModel(3, 25, adj, device).to(device)

	S_dataset, T_dataset = get_datasets()

	S_train, T_train, S_val, T_val = split_datasets(S_dataset, T_dataset)
# Change S_dataset and T_dataset here for training & validation
	train_loader = KimoreDataLoader(S_train, T_train, P=12, MB=1)
	val_loaders = KimoreValLoader(S_val, T_val)

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

	num_epochs = 1500

	# Get path for saving models
	models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

	for epoch in range(num_epochs):
		train_loss = train(model, train_loader, optimizer)
		val_loss = validate(model, val_loaders)

		if val_loss < 1.4:
			torch.save(model.state_dict(), os.path.join(models_path,f"STGCN_E{epoch+1}_T{train_loss:.4f}_V{val_loss:.4f}.pth"))

		print("Epoch: ", epoch, " - Training Loss: ", train_loss, " - Validation Loss: ", val_loss, flush = True)

def train(model, train_loader, optimizer):
	model.train()
	training_loss = 0.0
	counter = 0
	for joint_positions, labels in train_loader:
		# loss, auxillary_loss = model(joint_positions, labels)
		loss = model(joint_positions, labels)

		training_loss += loss
		counter += 1

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	return training_loss / counter

def validate(model, val_loaders):
	model.eval()
	validation_loss = 0.0
	counter = 0
	for val_loader in val_loaders:
		for joint_positions, labels in val_loader:
			loss = model.validate(joint_positions, labels)

			counter += 1
			validation_loss += loss
	return validation_loss / counter

if __name__ == '__main__':
	main()
