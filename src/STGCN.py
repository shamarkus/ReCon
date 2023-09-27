from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

class GCN(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(GCN, self).__init__()

		self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
		self.bias = nn.Parameter(torch.Tensor(out_channels))

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.weight)
		nn.init.zeros_(self.bias)

	def forward(self, inp):
		x, adj = inp
		# Calculate the degree matrix
		degree_matrix_inv_sqrt = torch.diag_embed((torch.sum(adj, dim=-1) + 1e-5) ** -0.5)

		# Normalize the adjacency matrix
		adj_norm = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, adj), degree_matrix_inv_sqrt)

		# Perform the graph convolution operation
		support = torch.matmul(x, self.weight)
		
		return torch.matmul(adj_norm, support) + self.bias

class GCNCONV(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(GCNCONV, self).__init__(node_dim=-2)
		self.conv = GCNConv(in_channels, out_channels)

	def forward(self, x, edge_index):
		return self.conv(x, edge_index)

# ResNet Bottleneck Architecture
# LSTM concatenated to the STGCN embedder --> TEMPORAL EMBEDDINGs 
#  f_q  -> LSTM -> g_q              T'xKxC where T' < T  (400, T' = 125) ( K < N, K = 6 ) 4x256 -> 4x32
class TemporalConvolutionModule(nn.Module):
  def __init__(self, out_channels, stride = (1,1), dropout = 0.2):
    super(TemporalConvolutionModule, self).__init__()
    self.tcn = nn.Sequential(
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=stride),
      nn.BatchNorm2d(out_channels),
      nn.Dropout(dropout))

  def forward(self, x):
    x = x.permute(0,3,1,2) # BxTxNxC to BxCxTxN
    return self.tcn(x).permute(0,2,3,1)

class SpatialTemporalLayer(nn.Module):
  def __init__(self, in_channels, out_channels, stride = (1,1), temporal_dropout=0.2):
    super(SpatialTemporalLayer, self).__init__()

    self.sl = GCNCONV(in_channels, out_channels)
    self.tl = TemporalConvolutionModule(out_channels, stride, temporal_dropout)

    if in_channels != out_channels or stride != (1,1):
      self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))
    else:
      self.residual = lambda x: x

    # self.norm = nn.LayerNorm(out_channels)
  def forward(self, inp):
    x, adj = inp
    so = self.sl(x, adj)
    return (F.relu(self.tl(so) + self.residual(x.permute(0,3,1,2)).permute(0,2,3,1)), adj)

class RotationInvariantDescriptor(nn.Module):
  def __init__(self, node_dim):
    super(RotationInvariantDescriptor, self).__init__()
    self.bn = nn.BatchNorm1d(node_dim ** 2)

  def forward(self, x):
    x = torch.einsum('btic,btjc->btij', x, x)

    # N and C will be the same in this case since we are doing dot product
    B, T, N, C = x.size()
    x = x.permute(0, 2, 3, 1).contiguous().view(B, N * C, T)
    x = self.bn(x)
    return x.view(B, N, C, T).permute(0, 3, 1, 2)

class DiffPool(nn.Module):
	def __init__(self, in_channels, num_clusters):
		super(DiffPool, self).__init__()

		# Spatial Pooling
		self.gnn_pool = GCNCONV(in_channels, num_clusters)
		# Temporal Pooling
		self.temporal_pool = lambda x: torch.mean(x, dim = 1)
		# Layer normalization
		self.norm = nn.LayerNorm(in_channels)

		self._S = None
	
	def forward(self, inp):
		x, adj, adj_dense = inp

		s = F.softmax(self.gnn_pool(x, adj), dim=-1)
		self._S = s

		z = self.norm(torch.einsum('btnk,btnc->btkc', s, x))
		coarse_adj = torch.einsum('btnk,ni,btij->btkj', s, adj_dense, s)

		return (self.temporal_pool(z), self.temporal_pool(coarse_adj))

class LSTMEmbedder(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers = 1):
		super(LSTMEmbedder, self).__init__()
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

	def forward(self, x):
		B, T, N, C = x.shape
		x = x.reshape(B, T, N * C)

		_, (h_n, _) = self.lstm(x)
		return h_n[-1]

class STEncoder(nn.Module):
  def __init__(self, input_dim, node_dim, num_clusters = 6, RI = True):
    super(STEncoder, self).__init__()

    if RI:
      self.ri = RotationInvariantDescriptor(node_dim)
      input_dim = node_dim
    else:
      self.ri = lambda x: x

    self.ste = nn.Sequential(
        SpatialTemporalLayer(input_dim, 64),
        SpatialTemporalLayer(64, 64),
        SpatialTemporalLayer(64, 128, stride=(2,1)),
        SpatialTemporalLayer(128, 128),
        SpatialTemporalLayer(128, 128),
        SpatialTemporalLayer(128, 256, stride=(2,1)),
        SpatialTemporalLayer(256, 256),
        SpatialTemporalLayer(256, 256))

    self.pool = DiffPool(256, num_clusters)

  def forward(self, x, adj, adj_dense):
    x = self.ri(x)

    x, _ = self.ste((x, adj))

    return self.pool((x, adj, adj_dense))

class STProjector(nn.Module):
	def __init__(self, in_channels = 256, num_clusters = 6, conv_dim = 48, out_channels = 256):
		super(STProjector, self).__init__()

		self.f = lambda x: torch.mean(x, dim = 1)
		self.h = nn.Sequential(
			GCN(in_channels, conv_dim), 
			nn.BatchNorm1d(num_clusters),
			nn.ReLU(),
			nn.Flatten())
		self.g = nn.Sequential(
			Linear(in_channels + num_clusters * conv_dim, out_channels),
			nn.ReLU(),
			Linear(out_channels, out_channels))

	def forward(self, inp):
		x, adj = inp

		# Generate ST body embeddings - Primary Outcome
		body_emb = self.f(x)

		# Generate ST reduced joint embeddings - Supplementary Controls
		joint_emb = self.h((x, adj))

		# Concatenate and pass through feed-forward MLP
		proj_emb = self.g(torch.cat((body_emb, joint_emb), dim = 1))

		# Normalize Embeddings
		return F.normalize(proj_emb, p = 2, dim = -1)
