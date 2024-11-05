import torch
from torch import nn
from torch_scatter.scatter import scatter
from e3nn.o3 import Linear


class AtomwiseLinear(nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 out_field='node_features'):
        
        super().__init__()
        self.irreps_out = irreps_out
        self.linear = Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.out_field = out_field

    def forward(self, data):
        if self.out_field != 'node_features':
            data[self.out_field] = self.linear(data['node_features'])
        else:
            data['node_features'] = self.linear(data['node_features'])
            data['node_features_irreps'] = self.irreps_out
        return data


class AtomwiseReduce(nn.Module):
    def __init__(self,
                 in_field='per_atom_pred',
                 out_field='pred'):
        
        super().__init__()
        self.in_field = in_field
        self.out_field = out_field
        
    def forward(self, data):
        nodes_to_mol_index = [mol_index for mol_index, num_atoms in enumerate(data['num_nodes']) for _ in range(num_atoms)]
        nodes_to_mol_index = torch.tensor(nodes_to_mol_index, dtype=torch.long, device=data[self.in_field].device)
        data[self.out_field] = scatter(data[self.in_field], nodes_to_mol_index, dim=0, reduce='sum').squeeze()
        return data
