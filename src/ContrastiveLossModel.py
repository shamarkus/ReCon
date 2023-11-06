import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R
from STGCNBaseline import STEncoder, STProjector
# from STGCN3RI import STEncoder, STProjector

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
		self.rot = Rotate(rot = 20)
		# self.shear = Shear()
		self.blur = GaussianBlur(sigma=[0.1, 0.4])
		self.noise = GaussianNoise(std=0.01)
		
	def __call__(self, x, T):
		x = self.temporal_crop(x, T)

		x = self.noise(x)
		x = self.blur(x)
		x = self.rot(x)
		# x = self.shear(x)

		return x

class ContrastiveLossModel(nn.Module):
	def __init__(self, input_dim, node_dim, adj, device, temperature = 0.1):
		super(ContrastiveLossModel, self).__init__()

		self.A = adj.detach().to(device).requires_grad_(False)
		self.__augment__ = Augmentor()

		self.cor_emb = torch.zeros(10,128).to(device)

		# Initialize the online and offline encoders
		self.f_q = STEncoder(input_dim, node_dim, RI = False)
		self.g_q = STProjector()

		# Hyperparameters
		self.temperature = temperature
		self.device = device
	def augment_samples(self, x_tensor):
		# # of frames to crop by
		_, T, _, _ = x_tensor.shape
		T = round(T * random.uniform(0.7, 1))

		# Apply __augment__ function on each sample
		return torch.stack([self.__augment__(sample, T) for sample in x_tensor], dim=0)

	def augment_and_transform(self, x):
		# Convert to two-view batch
		x_tensor = torch.cat([x,x])

		x_tensor = self.augment_samples(x_tensor)

		# Split tensor into two views: view1 and view2
		view1, view2 = x_tensor.chunk(2, dim=0)

		return view1.to(self.device), view2.to(self.device)

	def vectorized_cosine_similarity(self, emb):
		"""Compute the vectorized cosine similarity between all vectors in emb."""
		# emb is size M x P x d_k, output is M x P x P
		return torch.mm(emb,emb.T)
		# return torch.einsum('ic,jc->ij', emb, emb)

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
		B, C = q.shape

		# Flattened label & embedding tensors representative of 2-view batch (q || k)
		labels = torch.cat([labels, labels], dim = 0) # New label dimension should be 2B
		batch = torch.cat([q,k], dim = 0) # New batch dimension should be 2B x C

		I = torch.eye(2*B).to(self.device)
		zI = (torch.ones((2*B)).to(self.device) - I).to(self.device)

		# Similarity Matrix between embeddings: M x 2P x 2P
		sims = self.vectorized_cosine_similarity(batch) 
		# sims = self.vectorized_l1_similarity(batch) 
		sims = torch.exp((sims * zI) / self.temperature) - I

		# Label distance tensor: 2B x 2B
		dists = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(self.device)
		inc_mask = (labels <= -1).to(self.device)

		dists[:, inc_mask] = False;

		# P1 + P2 + ... + Pm / (A1 + A2 + ... + P1 + P2 + ... + An), size 2B
		sum2 = (dists * sims).sum(1) / sims.sum(1)

		cardinalP = dists.sum(1) - 1

		return torch.sum((-1 / cardinalP) * torch.log(sum2 + inc_mask))
	
	def get_norm(self, x, labels):
		labels = labels.to(self.device)
		x = x.to(self.device)

		q = self.f_q(x, self.A)
		q = self.g_q(q)

		self.cor_emb = self.cor_emb.index_add(0, labels, q)
		
	def apply_avg(self):
		self.cor_emb = F.normalize(self.cor_emb, p = 2, dim = -1)

	def validate(self, x, labels):
		labels = labels.to(self.device)
		x = x.to(self.device)

		# don't pass into projector
		q = self.f_q(x, self.A)
		q = self.g_q(q)

		master_emb = self.cor_emb[labels]
		return (master_emb * q).sum(dim = 1)
		
	def forward(self, x, labels):
		# Augment the data to get two views
		labels = labels.to(self.device)

		# view1 and view2 will both be B x T x N x C
		view1, view2 = self.augment_and_transform(x)

		q = self.f_q(view1, self.A)
		q = self.g_q(q)

		k = self.f_q(view2, self.A)
		k = self.g_q(k)

		# Compute the contrastive loss
		loss = self.contrastive_loss(q, k, labels)

		return loss 
