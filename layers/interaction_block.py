from typing import Optional
import torch
from torch import nn
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate

acts = {"abs": torch.abs,
        "tanh": torch.tanh,
        "silu": torch.nn.functional.silu}


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class InteractionBlock(nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 node_attrs_irreps,
                 edge_attrs_irreps,
                 num_basis: int = 8,
                 invariant_layers: int = 2,
                 invariant_neurons: int = 64,
                 use_sc: bool = True,
                 nonlinearity_scalars_act: str = "silu"):
        
        super().__init__()
        self.use_sc = use_sc
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.node_attrs_irreps = node_attrs_irreps
        self.edge_attrs_irreps = edge_attrs_irreps

        self.linear_1 = Linear(irreps_in=self.irreps_in,
                               irreps_out=self.irreps_in,
                               internal_weights=True,
                               shared_weights=True)

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.edge_attrs_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_in1, i_in2, p[i_out], mode, train) 
                        for i_in1, i_in2, i_out, mode, train in instructions]

        self.tp = TensorProduct(irreps_in1=self.irreps_in,
                                irreps_in2=self.edge_attrs_irreps,
                                irreps_out=irreps_mid,
                                instructions=instructions,
                                shared_weights=False,
                                internal_weights=False)

        hs = [num_basis] + invariant_layers * [invariant_neurons] + [self.tp.weight_numel]
        act = acts[nonlinearity_scalars_act]
        self.fc = FullyConnectedNet(hs=hs, act=act)

        self.linear_2 = Linear(irreps_in=irreps_mid.simplify(),
                               irreps_out=self.irreps_out,
                               internal_weights=True,
                               shared_weights=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in1=self.irreps_in,
                                                  irreps_in2=self.node_attrs_irreps,
                                                  irreps_out=self.irreps_out)
        else:
            self.sc = None

    def forward(self, data):
        weights = self.fc(data['edge_embeddings'])
        x = data['node_features']
        edge_src, edge_dst = data['edge_index']

        if self.sc is not None:
            sc = self.sc(x, data['node_attrs'])

        x = self.linear_1(x)
        edge_features = self.tp(x[edge_src], data['edge_attrs'], weights)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        x = x / (data['avg_num_neighbors'] ** 0.5)
  
        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        data['node_features'] = x
        return data


class ConvNetLayer(torch.nn.Module):
    def __init__(self,
                 feature_irreps_in,
                 feature_irreps_hidden,
                 edge_attrs_irreps,
                 convolution: torch.nn.Module = InteractionBlock,
                 convolution_kwargs: dict = {},
                 resnet: bool = True,
                 nonlinearity_scalars: dict = {'e': 'silu', 'o': 'tanh'},
                 nonlinearity_gates: dict = {'e': 'silu', 'o': 'tanh'}):

        super().__init__()
        self.irreps_prev_layer_out = feature_irreps_in
        self.irreps_hidden = feature_irreps_hidden
        self.edge_attrs_irreps = edge_attrs_irreps
        self.resnet = resnet

        nonlinearity_scalars = {1: nonlinearity_scalars["e"],
                                -1: nonlinearity_scalars["o"]}
        nonlinearity_gates = {1: nonlinearity_gates["e"],
                              -1: nonlinearity_gates["o"]}
        
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden
                                    if ir.l == 0 and tp_path_exists(self.irreps_prev_layer_out, self.edge_attrs_irreps, ir)])
        
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden
                                  if ir.l > 0 and tp_path_exists(self.irreps_prev_layer_out, self.edge_attrs_irreps, ir)])
        
        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        ir = "0e" if tp_path_exists(self.irreps_prev_layer_out, self.edge_attrs_irreps, "0e") else "0o"
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
        equivariant_nonlinearity = Gate(irreps_scalars=irreps_scalars,
                                        act_scalars=[acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars],
                                        irreps_gates=irreps_gates,
                                        act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                                        irreps_gated=irreps_gated)
        
        irreps_conv_out = equivariant_nonlinearity.irreps_in.simplify()
        self.equivariant_nonlinearity = equivariant_nonlinearity

        if irreps_layer_out == self.irreps_prev_layer_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        convolution_kwargs.update({"irreps_in": self.irreps_prev_layer_out,
                                   "irreps_out": irreps_conv_out})

        self.conv = convolution(**convolution_kwargs)
        self.irreps_layer_out = self.equivariant_nonlinearity.irreps_out

    def forward(self, data):
        old_x = data['node_features']
        data = self.conv(data)
        data['node_features'] = self.equivariant_nonlinearity(data['node_features'])

        if self.resnet:
            data['node_features'] = old_x + data['node_features']

        data['node_features_irreps'] = self.irreps_layer_out

        return data
    