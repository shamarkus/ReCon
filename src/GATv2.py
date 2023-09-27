from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
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

class GATv2Conv(MessagePassing):
  def __init__(self, in_channels, out_channels, heads = 1, negative_slope = 0.2, dropout = 0.):
    super(GATv2Conv, self).__init__(node_dim=-3)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.heads = heads
    self.negative_slope = negative_slope
    self.dropout = dropout

    self.lin_l = Linear(in_channels, heads * out_channels)
    self.lin_r = Linear(in_channels, heads * out_channels) # self.lin_r = self.lin_l

    self.att = Parameter(torch.Tensor(1, heads, out_channels))
    self.bias = Parameter(torch.Tensor(heads * out_channels))

    self.reset_parameters()

  def reset_parameters(self):
    glorot(self.lin_l.weight)
    glorot(self.lin_r.weight)
    glorot(self.att)
    zeros(self.bias)

  def forward(self, x, edge_index):
    # type: (Union[Tensor, PairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
    # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
    # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
    # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
    r"""
    Args:
        return_attention_weights (bool, optional): If set to :obj:`True`,
            will additionally return the tuple
            :obj:`(edge_index, attention_weights)`, holding the computed
            attention weights for each edge. (default: :obj:`None`)
    """
    B, T, N, C = x.shape
    H, C_ = self.heads, self.out_channels

    x_l = self.lin_l(x).view(B, T, N, H, C_)

    # Should we share weights?
    # x_r = x_l
    x_r = self.lin_r(x).view(B, T, N, H, C_)

    # propagate_type: (x: PairTensor)
    out = self.propagate(edge_index, x=(x_l, x_r), size=(N,N))

    out = out.contiguous().view(B, T, N, H * C_)

    out += self.bias

    return out

  def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
    x = x_i + x_j
    x = F.leaky_relu(x, self.negative_slope)
    alpha = (x * self.att).sum(dim=-1)
    alpha = softmax(alpha, index, ptr, size_i, dim = 2) # B x T x N x H where N is num messages
    print(alpha.transpose(-1, -2)[0][0][0])
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
    return x_j * alpha.unsqueeze(-1)

  def __repr__(self):
    return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class SpatialAttentionModule(nn.Module):
  def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.):
    super(SpatialAttentionModule, self).__init__()

    # Transformer-Style Multi-Head Attention
    self.gatv2 = GATv2Conv(in_channels, out_channels, heads, negative_slope=negative_slope, dropout=dropout)
    self.projection = Linear(out_channels * heads, out_channels)

    if in_channels != out_channels:
      self.residual = lambda x: nn.Conv1d(in_channels, out_channels, kernel_size=1).cuda()(x.reshape(-1,*x.shape[:-3:-1])).reshape(*x.shape[:-1], -1)
    else:
      self.residual = lambda x: x

    # Layer normalization
    self.norm = nn.LayerNorm(out_channels)

  def forward(self, x, adj):

    # Input: [B, T, N, C] - Apply non-linearity after GATv2 convolution
    x_mha = F.relu(self.gatv2(x, adj))

    # Input: [B, T, N, H, C_]
    x_sam = self.projection(x_mha) + self.residual(x)

    # Output: [B, T, N, C_]  ---- ReLu(LayerNorm(g(ReLu(f(x)))+x)) where g() is a projection, and f() is convolution
    return F.relu(self.norm(x_sam))

class TemporalConvolutionModule(nn.Module):
  def __init__(self, out_channels, stride = (1,1), dropout = 0.5):
    super(TemporalConvolutionModule, self).__init__()
    self.tcn = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=stride),
      # nn.BatchNorm2d(out_channels),
      nn.ReLU())
      # nn.Dropout(dropout))

  def forward(self, x):
    x = x.permute(0,3,1,2) # BxTxNxC to BxCxTxN
    return self.tcn(x).permute(0,2,3,1)

class SpatialTemporalLayer(nn.Module):
  def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0., stride = (1,1), temporal_dropout=0.5):
    super(SpatialTemporalLayer, self).__init__()

    # No resnet mechanism
    self.sl = SpatialAttentionModule(in_channels, out_channels, heads, negative_slope, dropout)
    self.tl = TemporalConvolutionModule(out_channels, stride, temporal_dropout)

  def forward(self, inp):
    x, adj = inp
    so = self.sl(x, adj)
    return (self.tl(so), adj)

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
		self.gnn_pool = SpatialAttentionModule(in_channels, num_clusters, 6)
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
    #self.ste = nn.Sequential(
    #    SpatialTemporalLayer(input_dim, 64, 6),
    #    SpatialTemporalLayer(64, 64, 6),
    #    SpatialTemporalLayer(64, 128, 6),
    #    SpatialTemporalLayer(128, 128, 6),
    #    SpatialTemporalLayer(128, 256, 6, stride=(2,1)),
    #    SpatialTemporalLayer(256, 256, 6),
    #    SpatialTemporalLayer(256, 512, 6),
    #    SpatialTemporalLayer(512, 512, 6, stride=(2,1))
    #)
    self.ste = nn.Sequential(
        SpatialTemporalLayer(input_dim, 32, 6),
        SpatialTemporalLayer(32, 32, 6),
        SpatialTemporalLayer(32, 64, 6),
        SpatialTemporalLayer(64, 64, 6),
        SpatialTemporalLayer(64, 128, 6, stride=(2,1)),
        SpatialTemporalLayer(128, 128, 6),
        SpatialTemporalLayer(128, 256, 6),
        SpatialTemporalLayer(256, 256, 6, stride=(2,1))
		)

    self.pool = DiffPool(256, num_clusters)


  def forward(self, x, adj, adj_dense):
    x = self.ri(x)

    x, _ = self.ste((x, adj))

    return x.mean(dim=1) # self.pool((x, adj, adj_dense))

class STProjector(nn.Module):
	def __init__(self, in_channels = 128, num_clusters = 6, conv_dim = 32, out_channels = 128):
		super(STProjector, self).__init__()

		self.f = lambda x: torch.mean(x, dim = 1)
		self.h = nn.Sequential(
			GCN(in_channels, conv_dim), 
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
