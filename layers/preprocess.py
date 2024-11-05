import pandas as pd
from rdkit import Chem
import numpy as np
import torch


def load_mol_graphs(path_to_docked, path_to_auc):
    mol_dataset = []
    data = pd.read_csv(path_to_auc)
    smiles = data['smiles']
    auc_values = data['AUC']
    auc_values = auc_values / auc_values.max() * 20.0
    
    for i in range(len(smiles)):
        sdf_file = f"{path_to_docked}/mol_{i+1}.sdf"
        try:
            mol = Chem.SDMolSupplier(sdf_file)[0]
        except OSError:
            continue
        
        molecule = {
            'mol': mol,
            'smiles': smiles[i],
            'auc': torch.tensor([auc_values[i]], dtype=torch.float)
        }
        
        molecule = preprocess_mol_dict(molecule)
        mol_dataset.append(molecule)
    
    atom_types = set()
    for mol in mol_dataset:
        atom_types.update(mol['atom_type_symbols'])
    atom_types = sorted(list(atom_types))
    atomtype2idx = {atomtype: idx for idx, atomtype in enumerate(atom_types)}

    for molecule in mol_dataset:
        molecule['atom_types'] = torch.tensor([atomtype2idx[atomtype] for atomtype in molecule['atom_type_symbols']], dtype=torch.long)

    return mol_dataset, len(atom_types)


def preprocess_mol_dict(mol_dict, atom_type="atomic_number"):
    mol = mol_dict['mol']
    mol_dict['num_nodes'] = len(mol.GetAtoms())

    if atom_type == "atomic_number":
        atom_type_symbols = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        mol_dict['atom_type_symbols'] = atom_type_symbols

    edges = np.stack(Chem.GetAdjacencyMatrix(mol).nonzero())
    mol_dict['edge_index'] = torch.tensor(edges, dtype=torch.long)
    edge_src, edge_dst = mol_dict['edge_index'][0], mol_dict['edge_index'][1]

    coords = mol.GetConformer().GetPositions()
    mol_dict['coords'] = torch.tensor(coords, dtype=torch.float)
    mol_dict['edge_vectors'] = mol_dict['coords'][edge_dst] - mol_dict['coords'][edge_src]
    mol_dict['edge_lengths'] = torch.norm(mol_dict['edge_vectors'], dim=1)

    del mol_dict['mol']
    return mol_dict

def calc_avg_num_neighbors(data):
    return len(data['edge_index'][0]) / sum(data['num_nodes'])
