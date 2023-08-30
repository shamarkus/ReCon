import random
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

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

    def create_batch(self, P, MB):
        # Initialize the big batch
        big_batch = []
        big_batch_labels = []

        for _ in range(MB):
            # Sample from exemplar dataset S
            s_sample_idx = random.choice(range(len(self.S)))
            s_sample, s_label = self.S[s_sample_idx]
            movement = self.S.data[s_sample_idx][1]

            s_2 = [s_sample]
            l_2 = [s_label]

            # Sample P-1 samples from non-exemplar dataset T with the same movement
            t_samples = [self.T[i] for i in range(len(self.T)) if self.T.data[i][1] == movement]
            t_samples = random.sample(t_samples, P-1)
            t_p = [sample[0] for sample in t_samples]
            l_p = [sample[1] for sample in t_samples]

            # Merge and add to big batch
            small_batch = s_2 + t_p
            small_labels = l_2 + l_p

            big_batch.append(small_batch)
            big_batch_labels.append(small_labels)

        # Augment the big batch to get the final 2-view batch
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
        self.idx_S = list(range(len(S_dataset)))
        self.idx_T = list(range(len(T_dataset)))
        random.shuffle(self.idx_S)
        random.shuffle(self.idx_T)

    def __iter__(self):
        self.current_S = 0
        self.current_T = 0
        return self

    def __next__(self):
        # If we've used up all the samples, stop the iteration
        if self.current_S >= len(self.idx_S) or self.current_T >= len(self.idx_T):
            raise StopIteration

        # Create a batch using the next set of indices
        batch = self.batch_creator.create_batch(self.P, self.MB)

        # Update the current indices
        self.current_S += self.MB
        self.current_T += (self.MB * (self.P - 1))

        return batch

    def __len__(self):
        # Return the number of batches this DataLoader will produce
        return min(len(self.S) // self.MB, len(self.T) // (self.MB * (self.P - 1)))


def createKimoreAdjacencyMatrix():
    connectivity = [(0, 1), (0, 12) , (0, 16) , (1, 20) , (12, 13) , (13, 14) , (14, 15) , (16, 17) , (17, 18) , (18, 19) , (20, 4) , (20, 8) , (20, 2) , (2, 3) , (4, 5) , (5, 6) , (6, 7) , (6, 22) , (7, 21) , (8, 9) , (9, 10) , (10, 11) , (10, 24) , (11, 23)]

    # Create source and target node lists
    src_nodes = [i for i, j in connectivity]
    tgt_nodes = [j for i, j in connectivity]

    # Include reversed connections to make the edges undirected
    src_nodes += tgt_nodes
    tgt_nodes += [i for i, j in connectivity]

    # Adding self-edges
    num_nodes = max(max(i, j) for i, j in connectivity) + 1
    for i in range(num_nodes):
      src_nodes.append(i)
      tgt_nodes.append(i)

    # Convert lists to tensors
    adj = torch.tensor([src_nodes, tgt_nodes])

    return adj

