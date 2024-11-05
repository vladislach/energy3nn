import torch
from torch.utils.data import Dataset
import numpy as np


class GraphDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    

def collate_fn(batch):
    out = {'num_nodes': [],
           'edge_index': [],
           'edge_lengths': [],
           'node_attrs': [],
           'node_features': [],
           'edge_attrs': [],
           'auc': []}
    
    index_shifts = np.cumsum([0] + [b['num_nodes'] for b in batch])[:-1]

    for i in range(len(batch)):
        mol_graph = batch[i]
        for key in mol_graph.keys():
            if key == 'edge_index':
                out[key].append(mol_graph[key] + index_shifts[i])
            elif key in out.keys():
                out[key].append(mol_graph[key])

    for key in out.keys():
        if key == 'num_nodes':
            pass
        elif key == 'edge_index':
            out[key] = torch.cat(out[key], dim=1)
        else:
            out[key] = torch.cat(out[key])

    out['node_attrs_irreps'] = batch[0]['node_attrs_irreps']
    out['node_features_irreps'] = batch[0]['node_features_irreps']
    out['edge_attrs_irreps'] = batch[0]['edge_attrs_irreps']

    return out
