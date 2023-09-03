import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg

from grit.layer.filered_layer import FilteredGINELayer
import torch_sparse


class HeatConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers):
        super().__init__()
        self.mask_encoder = RRWPMaskEncoderLayer()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        for layer in range(n_layers):
            if layer == 0:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters,
                    extended=cfg.gnn.MaskEncoder.extended
                ))
            else:
                self.layers.append(HeatConvLayer(
                    dim_in=dim_out,
                    dim_out=dim_out,
                    dropout=cfg.gnn.dropout,
                    train_eps=cfg.gnn.train_eps,
                    batch_norm=cfg.gnn.batchnorm,
                    residual=cfg.gnn.residual,
                    num_clusters=cfg.gnn.num_clusters,
                    extended=cfg.gnn.MaskEncoder.extended
                ))
            self.bns.append(nn.BatchNorm1d(dim_out))

    def forward(self, batch, cur_layer):
        self.mask_encoder(batch, cur_layer)
        for i in range(self.n_layers):
            self.layers[i](batch, cur_layer)
        return batch
                
class RRWPMaskEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        mask_cfg = cfg.gnn.MaskEncoder
        self.n_layers = mask_cfg.n_layers
        self.hidden_dim = mask_cfg.hidden_dim
        self.batch_norm = mask_cfg.batch_norm
        self.num_clusters = cfg.gnn.num_clusters
        self.dim_in = mask_cfg.ksteps
        
        self.bn = nn.ModuleList()
        self.model = nn.ModuleList()
        if self.n_layers == 1:
            self.model.append(nn.Linear(self.dim_in, self.num_clusters))
            self.bn.append(nn.BatchNorm1d(self.num_clusters))
        else:
            self.model.append(nn.Linear(self.dim_in, self.hidden_dim))
            self.bn.append(nn.BatchNorm1d(self.hidden_dim))
            for _ in range(self.n_layers - 2):
                self.model.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.bn.append(nn.BatchNorm1d(self.hidden_dim))
            self.model.append(nn.Linear(self.hidden_dim, self.num_clusters))
            self.bn.append(nn.BatchNorm1d(self.num_clusters))
        
    def forward(self, batch, cur_layer):
        if not (hasattr(batch, 'rrwp_val')):
            raise ValueError("Precomputed rrwp_val is "
                f"required for {self.__class__.__name__}; ")

        if not (hasattr(batch, 'extended_edge_index')):
            device = batch.x.device
            extended_edge_val_dummy = torch.zeros([batch.rrwp_index.shape[1], batch.edge_attr.shape[1]], dtype=torch.int64).to(device)
            extended_edge_index, extended_edge_val = torch_sparse.coalesce(torch.cat([batch.edge_index, batch.rrwp_index], dim=1), 
                    torch.cat([batch.edge_attr, extended_edge_val_dummy], dim=0), batch.num_nodes, batch.num_nodes,
                    op="add")
            setattr(batch, "extended_edge_index", extended_edge_index)
            setattr(batch, "extended_edge_attr", extended_edge_val)
        # N x N x K -> N x N x num_clusters

        encoding = batch.rrwp_val
        for i in range(self.n_layers):
            encoding = self.model[i](encoding)
            if self.batch_norm:
                encoding = self.bn[i](encoding)
            encoding = F.relu(encoding)
        raw_masks = F.softmax(encoding, dim=-1)
        masks = torch.transpose(raw_masks, 0, 1)
        cur_mask = "masks_" + str(cur_layer)
        setattr(batch, cur_mask, masks)
        return batch
        
class HeatConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_clusters, dropout, train_eps=True,
                 batch_norm=True, residual=True, extended=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.num_clusters = num_clusters
        self.extended = extended
        
        self.model = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()

        for cluster in range(num_clusters):
            if cluster == 0:
                mlp = nn.Sequential(pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
                                    pyg_nn.Linear(dim_out, dim_out))
                self.model.append(FilteredGINELayer(mlp, train_eps=train_eps))
            else:
                mlp = nn.Sequential(pyg_nn.Linear(dim_out, dim_out), nn.ReLU(),
                                    pyg_nn.Linear(dim_out, dim_out))
                self.model.append(FilteredGINELayer(mlp, train_eps=train_eps))
            self.bn.append(nn.BatchNorm1d(dim_out))
        
    def forward(self, batch, cur_layer):
        x_in = batch.x
        cur_mask = "masks_" + str(cur_layer)
        masks = getattr(batch, cur_mask)

        if self.extended:
            edge_index = batch.extended_edge_index
            edge_attr = batch.extended_edge_attr
        else:
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr


        for cluster in range(self.num_clusters):
            x_tmp = batch.x.clone()
            batch.x = self.model[cluster](batch.x, edge_index, edge_attr, masks[cluster]) + x_tmp
            if self.batch_norm:
                batch.x = self.bn[cluster](batch.x)
        
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        if self.residual:
            batch.x = x_in + batch.x # residual connection
        
        return batch
        
