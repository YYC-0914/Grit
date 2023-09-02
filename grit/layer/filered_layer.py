import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch import Tensor
from torch_geometric.typing import OptPairTensor
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg

class FilteredGINELayer(pyg_nn.conv.MessagePassing):
    """GINEConv Layer with Filtration implementation.

    Modified torch_geometric.nn.conv.GINECon layer to perform learned message scaling
    according to graph_encoding
    """
    
    def __init__(self, nn, eps=0., train_eps=False, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, mask=None, size=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, mask=mask)
        
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)
    
    def message(self, x_j, edge_attr, mask):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                    "match. Consider setting the 'edge_dim' "
                    "attribute of 'GINEConv'")
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        return torch.einsum('i,ij->ij', mask, (x_j + edge_attr).relu())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
        

        