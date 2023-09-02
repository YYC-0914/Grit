import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.register import register_network
from grit.layer.heat_mask_layer import HeatConvBlock


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        print("Dim_in: ", dim_in)
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('HMNModel')
class HMNModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        layers = []
        hidden_dim = cfg.gnn.dim_inner
        for layer in range(cfg.gnn.layers_block):
            if layer == 0:
                layers.append(HeatConvBlock(
                    dim_in=dim_in,
                    dim_out=hidden_dim,
                    n_layers=cfg.gnn.HeatConvBlock_n_layers
                ))
            else:
                layers.append(HeatConvBlock(
                    dim_in=hidden_dim,
                    dim_out=hidden_dim,
                    n_layers=cfg.gnn.HeatConvBlock_n_layers,
                ))
                
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
    
    def forward(self, batch):
        batch = self.encoder(batch)
        for cur_layer in range(cfg.gnn.layers_block):
            batch = self.layers[cur_layer](batch, cur_layer)
        batch = self.post_mp(batch)
        return batch