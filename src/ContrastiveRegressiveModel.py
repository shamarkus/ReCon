import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R
from STGCNBaseline import STEncoder, STProjector

class TemporalCrop(object):
	# Random temporal selection - 50% to 100% RI methodology - frozen frames
	def __call__(self, x, T):
		indx = torch.sort(torch.randperm(x.shape[0])[:T])[0]
		return x[indx]
		
class GaussianNoise(object):
	def __init__(self, mean=0, std=0.02):
		self.mean = mean
		self.std = std
	def __call__(self, x):
		T, N, C = x.shape
		noise = torch.normal(mean=self.mean, std=self.std, size=(T, N, C))
		return x + noise

class GaussianBlur(nn.Module):
	def __init__(self, channels=3, kernel=13, sigma=[0.1, 0.6]):
		super(GaussianBlur, self).__init__()

		self.channels = channels
		self.kernel = kernel
		self.min_max_sigma = sigma
		radius = int(kernel / 2)
		self.kernel_index = torch.arange(-radius, radius + 1).float()

	def __call__(self, x):
		sigma = torch.FloatTensor(1).uniform_(self.min_max_sigma[0], self.min_max_sigma[1]).item()
		blur_filter = torch.exp(-torch.pow(self.kernel_index, 2.0) / (2.0 * sigma ** 2))
		kernel = blur_filter.unsqueeze(0).unsqueeze(0).float()
		kernel = kernel.repeat(self.channels, 1, 1, 1) # 3,1,1,15
		self.weight = nn.Parameter(data=kernel, requires_grad=False)

		prob = np.random.random_sample()
		if prob < 0.5:
			x = x.transpose(0,2).unsqueeze(0)
			x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )), groups=self.channels)
			x = x.squeeze().transpose(0,2) 
		return x

class Shear(object):
	def __call__(self, x, shear_magnitude = 0.4):
        # Generate random shear values
		s1_list = (torch.rand(3) * 2 - 1) * shear_magnitude
		s2_list = (torch.rand(3) * 2 - 1) * shear_magnitude

		shr = torch.tensor([[1, s1_list[0], s2_list[0]], [s1_list[1], 1, s2_list[1]], [s1_list[2], s2_list[2], 1]])
		return torch.einsum('ij,kmj->kmi', shr, x)

class Rotate(object):
	def __init__(self, rot):
		self.aug_angle = rot

	def __call__(self, x):
		euler = (2 * np.random.rand(3) - 1) * (self.aug_angle / 180.) * np.pi
		rotation = R.from_euler('zxy', euler, degrees=False).as_matrix()

		rot = torch.from_numpy(rotation).float()
		return torch.einsum('ij,kmj->kmi', rot, x)
		
class Augmentor(object):
	def __init__(self):
		# Temporal Augmentations
		self.temporal_crop = TemporalCrop()

		# Spatial Augmentations
		self.rot = Rotate(rot = 15)
		self.shear = Shear()
		self.blur = GaussianBlur(sigma=[0.1, 0.4])
		self.noise = GaussianNoise(std=0.01)
		
	def __call__(self, x, T):
		x = self.temporal_crop(x, T)

		x = self.noise(x)
		x = self.blur(x)
		x = self.rot(x)
		# x = self.shear(x)

		return x

class ContrastiveRegressiveModel(nn.Module):
    # Initialized temperature as 2.0
	def __init__(self, input_dim, node_dim, adj, device, temperature = 1.0):
		super(ContrastiveRegressiveModel, self).__init__()

		self.A = adj.detach().to(device).requires_grad_(False)
		self.A_dense = self.normalize_adjacency_matrix(torch.sparse_coo_tensor(self.A, torch.ones(self.A.shape[-1]).to(device), (node_dim, node_dim)).to_dense()).to(device).requires_grad_(False)
		self.__augment__ = Augmentor()

		# Initialize the online and offline encoders
		self.f_q = STEncoder(input_dim, node_dim, RI = False)
		self.g_q = STProjector()

		# Hyperparameters
		self.temperature = temperature
		self.device = device

	def augment_samples(self, x_tensor):
		# # of frames to crop by
		_, T, _, _ = x_tensor.shape
		T = round(T * random.uniform(0.5, 1))

		# Apply __augment__ function on each sample
		return torch.stack([self.__augment__(sample, T) for sample in x_tensor], dim=0)

	def augment_and_transform(self, x):
		# Convert list of tensors to a single tensor with shape (2, MB*P, 735, 25, 3)
		x_tensor = torch.stack([torch.stack([torch.stack(p, dim=0) for p in mb], dim=0) for mb in x], dim=0)

		# Reshape tensor to shape (2*MB*P, 735, 25, 3)
		x_tensor = x_tensor.view(-1, *x_tensor.shape[3:])

		# Augment each sample
		x_tensor = self.augment_samples(x_tensor)

		# Split tensor into two views: view1 and view2
		view1, view2 = x_tensor.chunk(2, dim=0)

		return view1.to(self.device), view2.to(self.device)

	def vectorized_cosine_similarity(self, emb):
		"""Compute the vectorized cosine similarity between all vectors in emb."""
		# emb is size M x P x d_k, output is M x P x P
		return torch.einsum('mic,mjc->mij', emb, emb)

	def vectorized_l1_similarity(self, emb):
		"""Compute the negative L1 norm between tensors a and b."""
		# emb is size Mx B x d_k, output is M x B x B
		return -torch.cdist(emb, emb, p=1)

	def label_distance(self, labels):
		"""Compute the L1 distance between label distances within tensor labels."""
		# labels is size M x P, output is M x P x P
		labels_exp = labels.unsqueeze(-1)
		return torch.cdist(labels_exp, labels_exp, p=1)

	def contrastive_loss(self, q, k, labels):
# Extracting dimensions
		M, P, C = q.shape

		# Flattened label & embedding tensors representative of 2-view batch (q || k)
		labels = labels.permute(1,0,2).contiguous().view(M, -1) # New label dimension should be M x 2P 
		batch = torch.cat([q,k], dim = 1) # New batch dimension should be M x 2P x C

		I = torch.eye(2*P).to(self.device)
		zI = (torch.ones((2*P)).to(self.device) - I).to(self.device)

		# Label distance tensor: M x 2P x 2P
		dists = self.label_distance(labels)
		# Similarity Matrix between embeddings: M x 2P x 2P
		# sims = self.vectorized_cosine_similarity(batch) 
		sims = self.vectorized_l1_similarity(batch) 
		sims = torch.exp((sims * zI) / self.temperature) - I

		# Optimized broadcasting
		past_thresh = (dists.unsqueeze(2) >= dists.unsqueeze(3)).to(self.device)
		# Set the diagonal elements to zero
		diag_mask = torch.eye(past_thresh.shape[2], device=self.device).bool()
		past_thresh[:, diag_mask] = 0

		past_thresh = past_thresh.permute(0,1,3,2)
		sum3 = (sims.unsqueeze(-1) * past_thresh).sum(2) + I

		interim = torch.log(sims / sum3 + I)
		# loss = torch.sum(torch.log(sims / sum3 + I)) * (-1 / ( M * (2*P) * (2*P - 1)))
		loss = torch.sum(interim[:,0:1]) * (-1 / ( M * (P - 1))) + torch.sum(interim[:,P:P+1]) * (-1 / ( M * (P - 1)))

		return loss / 2
	
	def normalize_adjacency_matrix(self, adj):
		D = torch.diag(torch.sum(adj, dim=1))
		D_inv_sqrt = torch.pow(D, -0.5)
		D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0
		return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)

	def frobenius_norm(self, adj, s):
		s_T = torch.einsum('btic,btjc->btij', s, s)
		return torch.norm(adj - s_T, p = 'fro')

	def entropy(self, x):
		return -torch.sum(x * torch.log(x + 1e-9), dim = -1)

	def auxillary_loss(self, adj_dense, s_q, s_k):
		B, T, N, K = s_q.shape

		h_q = torch.mean(self.entropy(s_q))
		h_k = torch.mean(self.entropy(s_k))
		entropy_loss = (h_q + h_k) / 2

		fro_q = self.frobenius_norm(adj_dense, s_q) / (B * T)
		fro_k = self.frobenius_norm(adj_dense, s_k) / (B * T)
		pooling_loss = (fro_q + fro_k) / 2

		return entropy_loss + pooling_loss

	def shuffle_data_updated(self, x):
		"""Shuffle data along the batch dimension."""
# Get shuffle indices
		indices = torch.randperm(x.size(0))

# Shuffle data
		x = x[indices]

# Save the shuffle indices for unshuffling
		self.shuffle_indices = indices

		return x

	def unshuffle_data_updated(self, x):
		"""Unshuffle data to its original order."""
# Get inverse permutation
		_, unshuffle_indices = self.shuffle_indices.sort()

# Unshuffle data
		x = x[unshuffle_indices]

		return x

	def validate(self, x, labels):
		labels = labels.to(self.device).unsqueeze(0)
		x = x.to(self.device)

		# At this point, we are expecting x to be a BxTxNxC tensor
		q = self.f_q(x, self.A, self.A_dense)
		q = self.g_q(q).unsqueeze(0)

		# Extracting dimensions
		M, P, C = q.shape

		# Label dimension should be unchanged - 1 x P 
		I = torch.eye(P).to(self.device)
		zI = (torch.ones((P)).to(self.device) - I).to(self.device)

		# Label distance tensor: 1 x P x P
		dists = self.label_distance(labels)
		# Similarity Matrix between embeddings: 1 x P x P
		sims = self.vectorized_l1_similarity(q) 
		sims = torch.exp((sims * zI) / self.temperature) - I

		print(dists[:,0:1])
		print(sims[:,0:1])
		# Optimized broadcasting
		past_thresh = (dists.unsqueeze(2) >= dists.unsqueeze(3)).to(self.device)
		# Set the diagonal elements to zero
		diag_mask = torch.eye(past_thresh.shape[2], device=self.device).bool()
		past_thresh[:, diag_mask] = 0

		past_thresh = past_thresh.permute(0,1,3,2)
		sum3 = (sims.unsqueeze(-1) * past_thresh).sum(2) + I

		interim = torch.log(sims / sum3 + I)
		# loss = torch.sum(torch.log(sims / sum3 + I)) * (-1 / ( M * (P) * (P - 1)))
		loss = torch.sum(interim[:,0:1]) * (-1 / ( M * (P - 1))) 

		return loss

	def forward(self, x, labels):

# Augment the data to get two views
		labels = torch.tensor(labels).to(self.device)
		_ , MB, P = labels.shape

# view1 and view2 will both be (MB * P) x T x N x C
		view1, view2 = self.augment_and_transform(x)

# Pass through the encoder to get the query embeddings
		view1 =	self.shuffle_data_updated(view1)

		q = self.f_q(view1, self.A, self.A_dense)
		q = self.g_q(q)
		# s_q = self.f_q.pool._S

		q = self.unshuffle_data_updated(q).reshape(MB,P,-1)

# Pass through the encoder to get the key embeddings
		view2 =	self.shuffle_data_updated(view2)

		k = self.f_q(view2, self.A, self.A_dense)
		k = self.g_q(k)
		# s_k = self.f_q.pool._S

		k = self.unshuffle_data_updated(k).reshape(MB,P,-1)

# Compute the contrastive loss
		loss = self.contrastive_loss(q, k, labels)
		# auxillary_loss = self.auxillary_loss(self.A_dense, s_q, s_k)

		return loss # , auxillary_loss
