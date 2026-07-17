import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GraphConv(nn.Module):
    def __init__(
            self, 
            in_node_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            num_layers=5, 
            num_heads=1, 
            alpha=0.5, 
            dropout=0.0, 
            use_bn=True, 
            use_residual=True,
    ):
        super().__init__()

        self.in_node_channels = in_node_channels
        self.in_net_channels = in_net_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual

        self.activation = F.elu

        self.node_fcs = nn.ModuleList()
        self.net_fcs = nn.ModuleList()
        self.node_bns = nn.ModuleList()
        self.net_bns = nn.ModuleList()
        self.convs = nn.ModuleList()

        self.node_fcs.append(nn.Linear(in_node_channels, hidden_channels))
        self.net_fcs.append(nn.Linear(in_net_channels, hidden_channels))
        self.node_bns.append(nn.BatchNorm1d(hidden_channels))
        self.net_bns.append(nn.BatchNorm1d(hidden_channels))

        for i in range(num_layers):
            self.convs.append(
                gnn.HeteroConv({
                    ('node', 'out', 'net'): gnn.GATv2Conv(
                        hidden_channels, 
                        hidden_channels, 
                        edge_dim=edge_dim, 
                        heads=num_heads, 
                        dropout=dropout, 
                        add_self_loops=False, 
                        concat=False,
                    ),
                    ('net', 'in', 'node'): gnn.GATv2Conv(
                        hidden_channels, 
                        hidden_channels, 
                        edge_dim=edge_dim, 
                        heads=num_heads, 
                        dropout=dropout, 
                        add_self_loops=False, 
                        concat=False,
                    ),
                }, aggr='mean')
            )
            self.node_bns.append(nn.BatchNorm1d(hidden_channels))
            self.net_bns.append(nn.BatchNorm1d(hidden_channels))

    def reset_parameters(self):
        for fc in self.node_fcs:
            fc.reset_parameters()
        for fc in self.net_fcs:
            fc.reset_parameters()
        for bn in self.node_bns:
            bn.reset_parameters()
        for bn in self.net_bns:
            bn.reset_parameters()
        for hetero_conv in self.convs:
            for _, conv in hetero_conv.items():
                conv.reset_parameters()

    def forward(self, data, time_emb, node_emb=None, net_emb=None):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        if x_dict['node'].dim() == 2:
            x_dict['node'] = x_dict['node'].unsqueeze(0)
            x_dict['net'] = x_dict['net'].unsqueeze(0)

        node_seq = x_dict['node'].size(1)
        net_seq = x_dict['net'].size(1)

        x_dict['node'] = self.node_fcs[0](x_dict['node'])
        x_dict['net'] = self.net_fcs[0](x_dict['net'])

        if self.use_bn:
            x_dict['node'] = x_dict['node'].transpose(1, 2)
            x_dict['node'] = self.node_bns[0](x_dict['node']).transpose(1, 2)
            x_dict['net'] = x_dict['net'].transpose(1, 2)
            x_dict['net'] = self.net_bns[0](x_dict['net']).transpose(1, 2)
        x_dict['node'] = self.activation(x_dict['node'])
        x_dict['net'] = self.activation(x_dict['net'])
        x_dict['node'] = F.dropout(x_dict['node'], p=self.dropout, training=self.training)
        x_dict['net'] = F.dropout(x_dict['net'], p=self.dropout, training=self.training)

        prev_layer = x_dict

        for i, conv in enumerate(self.convs):
            time_emb_node = time_emb.unsqueeze(1).expand(-1, node_seq, -1)
            time_emb_net = time_emb.unsqueeze(1).expand(-1, net_seq, -1)
            prev_node = prev_layer["node"]
            prev_net = prev_layer["net"]

            x_dict['node'] = x_dict['node'] + time_emb_node
            if node_emb is not None:
                x_dict['node'] = x_dict['node'] + node_emb
            x_dict['net'] = x_dict['net'] + time_emb_net
            if net_emb is not None:
                x_dict['net'] = x_dict['net'] + net_emb

            x_dict['node'] = x_dict['node'].reshape(-1, x_dict['node'].size(-1))
            x_dict['net'] = x_dict['net'].reshape(-1, x_dict['net'].size(-1))

            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            if self.use_residual:
                x_dict['node'] = self.alpha * x_dict['node'] + (1 - self.alpha) * prev_node.reshape(-1, prev_node.size(-1))
                x_dict['net'] = self.alpha * x_dict['net'] + (1 - self.alpha) * prev_net.reshape(-1, prev_net.size(-1))
            if self.use_bn:
                x_dict['node'] = self.node_bns[i+1](x_dict['node'])
                x_dict['net'] = self.net_bns[i+1](x_dict['net'])
            x_dict['node'] = self.activation(x_dict['node'])
            x_dict['net'] = self.activation(x_dict['net'])
            x_dict['node'] = F.dropout(x_dict['node'], p=self.dropout, training=self.training)
            x_dict['net'] = F.dropout(x_dict['net'], p=self.dropout, training=self.training)

            x_dict['node'] = x_dict['node'].view(-1, node_seq, x_dict['node'].size(-1))
            x_dict['net'] = x_dict['net'].view(-1, net_seq, x_dict['net'].size(-1))
            prev_layer = x_dict

        return x_dict
    

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        x = x.float().unsqueeze(-1)
        sinusoid_inp = x * self.inv_freq
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb
    

class GraphPlacer(nn.Module):
    def __init__(
            self, 
            in_node_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            num_layers=3, 
            num_heads=4, 
            dropout=0., 
            use_residual=True, 
            use_bn=True, 
        ):
        super().__init__()

        self.in_node_channels = in_node_channels
        self.in_net_channels = in_net_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.time_feats = hidden_channels * 4
        self.num_layers = num_layers
        self.dropout = dropout

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmb(self.time_feats),
            nn.Linear(self.time_feats, self.time_feats),
            nn.SiLU(),
            nn.Linear(self.time_feats, self.hidden_channels)
        )

        self.graph_conv = GraphConv(
            in_node_channels, 
            in_net_channels, 
            hidden_channels, 
            edge_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout=dropout, 
            use_bn=use_bn, 
            use_residual=use_residual,
        )
        self.net_fc = nn.Linear(self.hidden_channels, 1)
        self.cell_fc = nn.Linear(self.hidden_channels, 2)

    def forward(self, data, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.time_embed(t)
        x = self.graph_conv(data, t_emb)

        x_net = x['net']
        x_cell = x['node']

        x_cell = self.cell_fc(x_cell)
        x_net = self.net_fc(x_net)

        return x_net, x_cell


