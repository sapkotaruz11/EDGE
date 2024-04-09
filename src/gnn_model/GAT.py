import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class RGAT(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dim,
        etypes,
        num_heads,
        num_hidden_layers=1,
        dropout=0,
    ):
        super(RGAT, self).__init__()
        self.rel_names = etypes
        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_dim, h_dim, norm="right", weight=False, bias=False
                )
                for rel in etypes
            }
        )
        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(
            RGATLayer(
                in_dim,
                h_dim,
                num_heads,
                self.rel_names,
                activation=F.relu,
                dropout=dropout,
                last_layer_flag=False,
            )
        )
        for i in range(num_hidden_layers):
            self.layers.append(
                RGATLayer(
                    h_dim * num_heads,
                    h_dim,
                    num_heads,
                    self.rel_names,
                    activation=F.relu,
                    dropout=dropout,
                    last_layer_flag=False,
                )
            )
        self.layers.append(
            RGATLayer(
                h_dim * num_heads,
                out_dim,
                num_heads,
                self.rel_names,
                activation=None,
                last_layer_flag=True,
            )
        )
        return

    def forward(self, hg, h_dict=None, embed=False, eweight=None, **kwargs):
        if embed:
            h_dict = self.conv(hg, h_dict)
            out_put = {}
            for n_type, h in h_dict.items():
                out_put[n_type] = h.flatten(1)
            return out_put
        if hasattr(hg, "ntypes"):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class RGATLayer(nn.Module):

    def __init__(
        self,
        in_feat,
        out_feat,
        num_heads,
        rel_names,
        activation=None,
        dropout=0.0,
        last_layer_flag=False,
        bias=True,
    ):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.last_layer_flag = last_layer_flag
        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATv2Conv(
                    in_feat,
                    out_feat,
                    num_heads=num_heads,
                    bias=bias,
                    allow_zero_in_degree=False,
                )
                for rel in rel_names
            }
        )

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            if self.last_layer_flag:
                h = h.mean(1)
            else:
                h = h.flatten(1)
            out_put[n_type] = h.squeeze()
        return out_put
