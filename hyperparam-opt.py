import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from e3nn import o3
import numpy as np
import os

from layers.preprocess import load_mol_graphs
from layers.embedding import OneHotAtomEncoding, SphericalHarmonicEdgeAttrs
from layers.dataset import GraphDataset, collate_fn
from layers.model import NequIP


hyperparams = {'LMAX': [0, 1, 2, 3, 4],
               'NUM_BASIS': [6, 8, 10],
               'BNI': [8, 16, 32, 64],
               'INVARIANT_LAYERS': [1, 2, 3],
               'INVARIANT_NEURONS': [16, 32, 64, 128],
               'NUM_CONV_LAYERS': [1, 2, 3, 4, 5],
               'RESNET': [True, False],
               'USE_SC': [True, False],
               'BATCH_SIZE': [2, 5, 10],
               'LEARNING_RATE': [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]}

default_params = {'LMAX': 2,
                  'NUM_BASIS': 8,
                  'BNI': 32,
                  'INVARIANT_LAYERS': 2,
                  'INVARIANT_NEURONS': 64,
                  'NUM_CONV_LAYERS': 4,
                  'RESNET': False,
                  'USE_SC': True,
                  'BATCH_SIZE': 5,
                  'LEARNING_RATE': 1e-3}


for hp_name in hyperparams.keys():
    for hp_val in hyperparams[hp_name]:
        params = default_params.copy()
        params[hp_name] = hp_val

        LMAX = params['LMAX']
        RMAX = 2.0
        NUM_BASIS = params['NUM_BASIS']
        P = 6

        BASIS_KWARGS = {'r_max': RMAX, 'num_basis': NUM_BASIS, 'trainable': True}
        CUTOFF_KWARGS = {'r_max':RMAX, 'p': P}

        BNI = params['BNI']
        CHEMICAL_EMBEDDING_IRREPS_OUT = o3.Irreps(f"{BNI}x0e")
        FEATURE_IRREPS_HIDDEN = o3.Irreps([(BNI, (l, p)) for l in range(LMAX + 1) for p in [-1, 1]])
        CONV_TO_OUTPUT_HIDDEN_IRREPS_OUT = o3.Irreps(f"{int(BNI / 2)}x0e")

        NONLINEARITY_SCALARS = {'e': 'silu', 'o': 'tanh'}
        NONLINEARITY_GATES = {'e': 'silu', 'o': 'tanh'}

        INVARIANT_LAYERS = params['INVARIANT_LAYERS']
        INVARIANT_NEURONS = params['INVARIANT_NEURONS']

        NUM_CONV_LAYERS = params['NUM_CONV_LAYERS']
        RESNET = params['RESNET']
        USE_SC = params['USE_SC']
        PEP_ATOM_OUT_FIELD = 'per_atom_pred'
        PRED_OUT_FIELD = 'auc_pred'

        config = {'basis_kwargs': BASIS_KWARGS,
                  'cutoff_kwargs': CUTOFF_KWARGS,
                  'chemical_embedding_irreps_out': CHEMICAL_EMBEDDING_IRREPS_OUT,
                  'num_conv_layers': NUM_CONV_LAYERS,
                  'feature_irreps_hidden': FEATURE_IRREPS_HIDDEN,
                  'resnet': RESNET,
                  'nonlinearity_scalars': NONLINEARITY_SCALARS,
                  'nonlinearity_gates': NONLINEARITY_GATES,
                  'num_basis': NUM_BASIS,
                  'invariant_layers': INVARIANT_LAYERS,
                  'invariant_neurons': INVARIANT_NEURONS,
                  'use_sc': USE_SC,
                  'conv_to_output_hidden_irreps_out': CONV_TO_OUTPUT_HIDDEN_IRREPS_OUT,
                  'per_atom_out_field': PEP_ATOM_OUT_FIELD,
                  'pred_out_field': PRED_OUT_FIELD}
        
        BATCH_SIZE = params['BATCH_SIZE']
        TEST_SIZE = 0.2
        LEARNING_RATE = params['LEARNING_RATE']
        NUM_EPOCHS = 50

        path_to_ligands = 'data/ligands'
        path_to_auc = 'data/auc_scores.csv'
        mol_dataset, num_atom_types = load_mol_graphs(path_to_ligands, path_to_auc)

        for mol_dict in mol_dataset:
            mol_dict = OneHotAtomEncoding(num_types=num_atom_types)(mol_dict)
            mol_dict = SphericalHarmonicEdgeAttrs(l_max=LMAX)(mol_dict)

        train_data, test_data = train_test_split(mol_dataset, test_size=TEST_SIZE, random_state=2115)
        train_dataset, test_dataset = GraphDataset(train_data), GraphDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NequIP(**config, batch=next(iter(train_loader)))
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        loss_fn = torch.nn.MSELoss()

        train_losses, test_losses = [], []

        for epoch in range(NUM_EPOCHS):
            epoch_train_loss, epoch_test_loss = 0.0, 0.0

            model.train()
            for batch in train_loader:
                for key in batch:
                    batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

                optimizer.zero_grad()
                batch = model(batch)
                loss = loss_fn(batch['auc_pred'], batch['auc'])
                loss.backward()
                optimizer.step()       

                epoch_train_loss += loss.item() * len(batch['num_nodes'])

            model.eval()
            for batch in test_loader:
                for key in batch:
                    batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]

                batch = model(batch)
                loss = loss_fn(batch['auc_pred'], batch['auc'])
                
                epoch_test_loss += loss.item() * len(batch['num_nodes'])

            scheduler.step()

            train_losses.append(epoch_train_loss / len(train_loader.dataset))
            test_losses.append(epoch_test_loss / len(test_loader.dataset))

        model.eval()
        y_true, y_pred = [], []

        for batch in test_loader:
            for key in batch:
                batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
            batch = model(batch)
            y_true.append(batch['auc'].detach().cpu().numpy())
            y_pred.append(batch['auc_pred'].detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        

        directory_name = f"model_{hp_name}_{hp_val}"
        output_dir = os.path.join('outputs', directory_name)
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

        np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(output_dir, 'test_losses.npy'), np.array(test_losses))

        np.save(os.path.join(output_dir, 'y_true.npy'), y_true)
        np.save(os.path.join(output_dir, 'y_pred.npy'), y_pred)
