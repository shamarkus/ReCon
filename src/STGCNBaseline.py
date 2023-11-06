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
		
		res = torch.matmul(adj_norm, support) + self.bias

		return res.transpose(-1,-2)

class GCNCONV(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(GCNCONV, self).__init__(node_dim=-2)
		self.conv = GCNConv(in_channels, out_channels)

	def forward(self, x, edge_index):
		return self.conv(x, edge_index)

# ResNet Bottleneck Architecture
class TemporalConvolutionModule(nn.Module):
  def __init__(self, out_channels, stride = (1,1), h = 2):
    super(TemporalConvolutionModule, self).__init__()
    self.tcn = nn.Sequential(
      nn.BatchNorm2d(out_channels),
      nn.ReLU(), 
      nn.Conv2d(out_channels, out_channels // h, kernel_size=1, stride=(1,1)),
      nn.BatchNorm2d(out_channels // h),
      nn.ReLU(), 
      nn.Conv2d(out_channels // h, out_channels // h, kernel_size=(9, 1), padding=(4, 0), stride=stride),
      nn.BatchNorm2d(out_channels // h),
      nn.ReLU(), 
      nn.Conv2d(out_channels // h, out_channels, kernel_size=1, stride=(1,1)),
      nn.BatchNorm2d(out_channels))

  def forward(self, x):
    x = x.permute(0,3,1,2) # BxTxNxC to BxCxTxN
    return self.tcn(x).permute(0,2,3,1)

class SpatialTemporalLayer(nn.Module):
  def __init__(self, in_channels, out_channels, stride = (1,1), h = 2):
    super(SpatialTemporalLayer, self).__init__()

    self.sl = GCNCONV(in_channels, out_channels)
    self.tl = TemporalConvolutionModule(out_channels, stride, h)

    if in_channels != out_channels or stride != (1,1):
      self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))
    else:
      self.residual = lambda x: x

    # self.norm = nn.LayerNorm(out_channels)
  def forward(self, inp):
    x, adj = inp
    so = self.sl(x, adj)
    return (F.relu(self.tl(so) + self.residual(x.permute(0,3,1,2)).permute(0,2,3,1)), adj)

class NonRotationInvariantDescriptor(nn.Module):
	def __init__(self, input_dim, node_dim):
		super(NonRotationInvariantDescriptor, self).__init__()
		self.bn = nn.BatchNorm1d(input_dim * node_dim)

	def forward(self, x):
		B, T, N, C = x.size()
		x = x.permute(0, 2, 3, 1).contiguous().view(B, N * C, T)
		x = self.bn(x)
		return x.view(B, N, C, T).permute(0, 3, 1, 2)

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
  def __init__(self, input_dim, node_dim, num_clusters = 6, RI = False):
    super(STEncoder, self).__init__()

    if RI:
      self.ri = RotationInvariantDescriptor(node_dim)
      input_dim = node_dim
    else:
      self.ri = NonRotationInvariantDescriptor(input_dim, node_dim)

    self.ste = nn.Sequential(
        SpatialTemporalLayer(input_dim, 64),
        SpatialTemporalLayer(64, 64),
        SpatialTemporalLayer(64, 128, stride=(2,1)),
        SpatialTemporalLayer(128, 128),
        SpatialTemporalLayer(128, 128),
        SpatialTemporalLayer(128, 256, stride=(2,1)),
        SpatialTemporalLayer(256, 256),
        SpatialTemporalLayer(256, 256),)

    self.pool = lambda x: torch.mean(x, dim = [1,2])

  def forward(self, x, adj):
    x = self.ri(x)

    x, _ = self.ste((x, adj))

    return self.pool(x)

class STProjector(nn.Module):
	def __init__(self, in_channels = 256, out_channels = 128):
		super(STProjector, self).__init__()

		self.g = nn.Sequential(
			Linear(in_channels, in_channels),
			nn.ReLU(),
			Linear(in_channels, out_channels))

	def forward(self, x):
		proj_emb = self.g(x)

		# Normalize Embeddings
		return F.normalize(proj_emb, p = 2, dim = -1)
