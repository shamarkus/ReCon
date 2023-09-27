import random
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader

class KimoreCustomDataset(Dataset):
    def __init__(self, data):
        self.K = 15
        if(data[0][0].shape[-1] != 3):
          for i in data[:, 0:1]:
              i[0] = i[0][:, :-1].reshape((-1,25,4))[:,:,:3]

          self.data = np.concatenate((data[:, 0:1], data[:, 4:5], data[:, 3:4], data[:, 2:3]), axis=1)
        else:
          self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions, movement, label, group = self.data[idx]

        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return joint_positions, label

    def filter_by_exercise(self, exercise):
        filtered_data = [data for data in self.data if data[1] == exercise]
        return KimoreCustomDataset(filtered_data)

    # K-top partition, class-balanced
    def filter_by_exemplar_group(self, exemplar=True):
        movement_to_samples = defaultdict(list)
        for sample in self.data:
            movement = sample[1]
            label = sample[2]
            movement_to_samples[movement].append((sample, label))

        if exemplar:
            filtered_data = []
            for movement, samples in movement_to_samples.items():
                sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)[:self.K]
                filtered_data.extend([sample[0] for sample in sorted_samples])
        else:
            filtered_data = []
            for movement, samples in movement_to_samples.items():
                sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)[self.K:]
                filtered_data.extend([sample[0] for sample in sorted_samples])

        return KimoreCustomDataset(filtered_data)
    def get_subset(self, indices):
        return KimoreCustomDataset([self.data[i] for i in indices])

    def append(self, new_data):
        self.data.extend(new_data)

class KimoreBatchCreatorUpdated:
    def __init__(self, S_dataset, T_dataset):
        self.S = S_dataset
        self.T = T_dataset
        
        # Group by movement
        self.grouped_S = self.group_by_movement(S_dataset)
        self.grouped_T = self.group_by_movement(T_dataset)
        
        # Keep track of indices already used for each movement in T_dataset
        self.used_indices_T = {movement: [] for movement in self.grouped_T.keys()}

    def group_by_movement(self, dataset):
        grouped_data = {}
        for i, (joint_positions, label) in enumerate(dataset):
            movement = dataset.data[i][1]
            if movement not in grouped_data:
                grouped_data[movement] = []
            grouped_data[movement].append((i, joint_positions, label))
        return grouped_data

    def create_batch(self, P, MB):
        big_batch = []
        big_batch_labels = []

        for _ in range(MB):
            # Sequentially sample from exemplar dataset S
            if not self.grouped_S:
                # S_dataset exhausted
                break
                
            movement, s_samples = list(self.grouped_S.items())[0]
            
            s_sample_idx, s_sample, s_label = s_samples.pop(0)
            if not s_samples:
                # If all samples of this movement from S_dataset have been used, remove it from the group.
                del self.grouped_S[movement]

            s_2 = [s_sample]
            l_2 = [s_label]
            
            # Sample P-1 samples from non-exemplar dataset T with the same movement
            available_samples = [item for item in self.grouped_T[movement] if item[0] not in self.used_indices_T[movement]]
            if len(available_samples) < P-1:
                # If we don't have enough fresh T_dataset samples, get some already used ones
                already_used_samples = [item for item in self.grouped_T[movement] if item[0] in self.used_indices_T[movement]]
                t_samples = random.sample(available_samples + already_used_samples, P-1)
            else:
                t_samples = random.sample(available_samples, P-1)
                
            for idx, _, _ in t_samples:
                self.used_indices_T[movement].append(idx)
            
            t_p = [sample[1] for sample in t_samples]
            l_p = [sample[2] for sample in t_samples]

            # Merge and add to big batch
            small_batch = s_2 + t_p
            small_labels = l_2 + l_p

            big_batch.append(small_batch)
            big_batch_labels.append(small_labels)

        final_batch = [big_batch, big_batch]
        final_batch_labels = [big_batch_labels, big_batch_labels]

        return final_batch, final_batch_labels

class KimoreDataLoader:
    def __init__(self, S_dataset, T_dataset, P, MB):
        self.S = S_dataset
        self.T = T_dataset
        self.P = P
        self.MB = MB
        self.batch_creator = KimoreBatchCreatorUpdated(S_dataset, T_dataset)

    def __iter__(self):
        self.batch_creator.used_indices_T = {movement: [] for movement in self.batch_creator.grouped_T.keys()}
        self.batch_creator.grouped_S = self.batch_creator.group_by_movement(self.S)
        return self
    def __next__(self):
        # If we've used up all the samples in S_dataset, stop the iteration
        if not self.batch_creator.grouped_S:
            raise StopIteration

        return self.batch_creator.create_batch(self.P, self.MB)

    def __len__(self):
        # The length of the DataLoader is essentially how many times we can create a batch.
        # This is determined by the length of the S_dataset divided by MB
        return len(self.S) // self.MB

def KimoreValLoader(S_val, T_val):
# Combine the two validation datasets
	combined_data = S_val.data + T_val.data

# Split the combined data based on exercise
	movements = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']
	dataloaders = []

	for movement in movements:
		filtered_data = [sample for sample in combined_data if sample[1] == movement]
		filtered_dataset = KimoreCustomDataset(filtered_data)

# Create a DataLoader with all samples in a single batch
		dataloader = DataLoader(filtered_dataset, batch_size=len(filtered_dataset), shuffle=False)
		dataloaders.append(dataloader)

	return dataloaders

def createKimoreAdjacencyMatrix():
    connectivity = [(0, 1), (0, 12) , (0, 16) , (1, 20) , (12, 13) , (13, 14) , (14, 15) , (16, 17) , (17, 18) , (18, 19) , (20, 4) , (20, 8) , (20, 2) , (2, 3) , (4, 5) , (5, 6) , (6, 7) , (6, 22) , (7, 21) , (8, 9) , (9, 10) , (10, 11) , (10, 24) , (11, 23)]

    # Create source and target node lists
    src_nodes = [i for i, j in connectivity]
    tgt_nodes = [j for i, j in connectivity]

    # Include reversed connections to make the edges undirected
    src_nodes += tgt_nodes
    tgt_nodes += [i for i, j in connectivity]

    # Adding self-edges
#     num_nodes = max(max(i, j) for i, j in connectivity) + 1
#     for i in range(num_nodes):
#       src_nodes.append(i)
#       tgt_nodes.append(i)

    # Convert lists to tensors
    adj = torch.tensor([src_nodes, tgt_nodes])

    return adj

