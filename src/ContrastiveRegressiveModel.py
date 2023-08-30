import torch
import torch.nn as nn
import torch.nn.functional as F

from GATv2 import STEncoder, STProjector

class ContrastiveRegressiveModel(nn.Module):
    # Initialized temperature as 2.0
	def __init__(self, input_dim, node_dim, adj, device, temperature = 0.10):
		super(ContrastiveRegressiveModel, self).__init__()

		self.A = adj.detach().to(device).requires_grad_(False)
		self.A_dense = self.normalize_adjacency_matrix(torch.sparse_coo_tensor(self.A, torch.ones(self.A.shape[-1]).to(device), (node_dim, node_dim)).to_dense()).to(device).requires_grad_(False)

# Initialize the online and offline encoders
		self.f_q = STEncoder(input_dim, node_dim, RI = False)
		self.g_q = STProjector()

# Hyperparameters
		self.temperature = temperature
		self.device = device

	def __augment__(self, x):
        # TODO
        # Rotation x, y, z
        # Shear
        # Temporal Crop
        # Temporal flip (talk to ali about this more)
        # Gaussian noise - very minimal
		return x

	# Everything but augmentation should be done beforehand 
	def augment_and_transform(self, x):
# Convert list of tensors to a single tensor with shape (2, MB*P, 735, 25, 3)
		x_tensor = torch.stack([torch.stack([torch.stack(p, dim=0) for p in mb], dim=0) for mb in x], dim=0)

# Reshape tensor to shape (2*MB*P, 735, 25, 3)
		x_tensor = x_tensor.view(-1, *x_tensor.shape[3:])

# Apply __augment__ function on each sample
		x_tensor = torch.stack([self.__augment__(sample) for sample in x_tensor], dim=0)

# Split tensor into two views: view1 and view2
		view1, view2 = x_tensor.chunk(2, dim=0)

		return view1.to(self.device), view2.to(self.device)

	def vectorized_cosine_similarity(self, emb):
		"""Compute the vectorized cosine similarity between all vectors in emb."""
		# emb is size M x P x d_k, output is M x P x P
		return torch.einsum('mic,mjc->mij', emb, emb)

	def vectorized_l2_similarity(self, emb):
		"""Compute the negative L2 norm between tensors a and b."""
		# emb is size B x d_k, output is B x B
		return -torch.cdist(emb, emb, p=2)

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

		# This is weird... Need to confirm why this is here with Ali
		I = torch.eye(2*P).to(self.device)
		zI = (torch.ones((2*P)).to(self.device) - I).to(self.device)

		# Label distance tensor: M x 2P x 2P
		dists = self.label_distance(labels)
		# Similarity Matrix between embeddings: M x 2P x 2P
		sims = self.vectorized_cosine_similarity(batch) 
		sims = torch.exp((sims * zI) / self.temperature) - I

		# Optimized broadcasting
		past_thresh = (dists.unsqueeze(2) >= dists.unsqueeze(3)).to(self.device)
		# Set the diagonal elements to zero
		diag_mask = torch.eye(past_thresh.shape[2], device=self.device).bool()
		past_thresh[:, diag_mask] = 0

		past_thresh = past_thresh.permute(0,1,3,2)
		sum3 = (sims.unsqueeze(-1) * past_thresh).sum(2) + I

		print(dists, sims)
		loss = torch.sum(torch.log(sims / sum3 + I)) * (-1 / ( M * (2*P) * (2*P - 1)))
		return loss
	
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
		s_q = self.f_q.pool._S

		q = self.unshuffle_data_updated(q).reshape(MB,P,-1)

# Pass through the encoder to get the key embeddings
		view2 =	self.shuffle_data_updated(view2)

		k = self.f_q(view2, self.A, self.A_dense)
		k = self.g_q(k)
		s_k = self.f_q.pool._S

		k = self.unshuffle_data_updated(k).reshape(MB,P,-1)

# Compute the contrastive loss
		loss = self.contrastive_loss(q, k, labels)
		auxillary_loss = self.auxillary_loss(self.A_dense, s_q, s_k)

		# print(loss, auxillary_loss)
		# return loss + auxillary_loss
		return loss
