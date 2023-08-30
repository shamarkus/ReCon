import pickle
import numpy as np
import torch

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
  avg_length = int(np.mean(all_lengths))

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

def main():
    # Create an instance of the custom DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj = createKimoreAdjacencyMatrix()

    model = ContrastiveRegressiveModel(3, 25, adj, device).to(device)

    S_dataset, T_dataset = get_datasets()
    kimore_loader = KimoreDataLoader(S_dataset, T_dataset, P=3, MB=1)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

    for joint_positions, labels in kimore_loader:
        break
    # Sample: How to use it in a training loop
    num_epochs = 1500
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Your training code here
        # model, optimizer, loss_function, etc.
        
        loss = model(joint_positions, labels)
        
        running_loss = running_loss + loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch ", epoch, flush = True)
        print("Loss ", running_loss, flush = True)

    # Return the number of batches this DataLoader will produce in one epoch

if __name__ == '__main__':
    main()
