from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.MaskEncoder = CN()
    cfg.gnn.MaskEncoder.mask_type = "RRWP"
    cfg.gnn.MaskEncoder.n_layers = 2
    cfg.gnn.MaskEncoder.hidden_dim = 16
    cfg.gnn.MaskEncoder.batch_norm = True
    cfg.gnn.MaskEncoder.ksteps = 21
    cfg.gnn.MaskEncoder.add_identity = True
    cfg.gnn.MaskEncoder.add_node_attr = False
    cfg.gnn.MaskEncoder.add_inverse= False
    cfg.gnn.MaskEncoder.extended = True


    cfg.gnn.num_clusters = 3
    cfg.gnn.train_eps = False
    cfg.gnn.HeatConvBlock_n_layers = 1
    cfg.gnn.layers_block = 5
