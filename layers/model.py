from torch import nn
from e3nn import o3

from .preprocess import calc_avg_num_neighbors
from .embedding import RadialBasisEmbedding
from .atomwise import AtomwiseLinear, AtomwiseReduce
from .interaction_block import ConvNetLayer


class NequIP(nn.Module):
    def __init__(self,
                 basis_kwargs,
                 cutoff_kwargs,
                 chemical_embedding_irreps_out,
                 num_conv_layers,
                 feature_irreps_hidden,
                 resnet,
                 nonlinearity_scalars,
                 nonlinearity_gates,
                 num_basis,
                 invariant_layers,
                 invariant_neurons,
                 use_sc,
                 conv_to_output_hidden_irreps_out,
                 per_atom_out_field,
                 pred_out_field,
                 batch):
        super().__init__()

        self.radial_basis_embedding = RadialBasisEmbedding(basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs)
        batch = self.radial_basis_embedding(batch)
        self.chemical_embedding = AtomwiseLinear(irreps_in=batch['node_features_irreps'], irreps_out=chemical_embedding_irreps_out)
        batch = self.chemical_embedding(batch)
        batch['avg_num_neighbors'] = calc_avg_num_neighbors(batch)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            conv_layer = ConvNetLayer(feature_irreps_in=batch['node_features_irreps'],
                                      feature_irreps_hidden=feature_irreps_hidden,
                                      edge_attrs_irreps=batch['edge_attrs_irreps'],
                                      resnet=resnet,
                                      nonlinearity_scalars=nonlinearity_scalars,
                                      nonlinearity_gates=nonlinearity_gates,
                                      convolution_kwargs={'irreps_in': batch['node_features_irreps'],
                                                          'irreps_out': batch['node_features_irreps'],
                                                          'node_attrs_irreps': batch['node_attrs_irreps'],
                                                          'edge_attrs_irreps': batch['edge_attrs_irreps'],
                                                          'num_basis': num_basis,
                                                          'invariant_layers': invariant_layers,
                                                          'invariant_neurons': invariant_neurons,
                                                          'use_sc': use_sc,
                                                          'nonlinearity_scalars_act': 'silu'})
            batch = conv_layer(batch)
            self.conv_layers.append(conv_layer)

        self.conv_to_output_hidden = AtomwiseLinear(irreps_in=batch['node_features_irreps'], irreps_out=conv_to_output_hidden_irreps_out)
        batch = self.conv_to_output_hidden(batch)
        self.output_hidden_to_scalar = AtomwiseLinear(irreps_in=batch['node_features_irreps'],
                                                      irreps_out=o3.Irreps('1x0e'),
                                                      out_field=per_atom_out_field)
        batch = self.output_hidden_to_scalar(batch)
        self.atomwise_reduce = AtomwiseReduce(in_field=per_atom_out_field, out_field=pred_out_field)
        batch = self.atomwise_reduce(batch)
    
    def forward(self, data):
        data = self.radial_basis_embedding(data)
        data = self.chemical_embedding(data)
        data['avg_num_neighbors'] = calc_avg_num_neighbors(data)

        for conv_layer in self.conv_layers:
            data = conv_layer(data)

        data = self.conv_to_output_hidden(data)
        data = self.output_hidden_to_scalar(data)
        data = self.atomwise_reduce(data)
        return data
