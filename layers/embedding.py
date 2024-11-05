import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode
import math


class OneHotAtomEncoding(nn.Module):
    def __init__(self,
                 num_types: int):
        super().__init__()
        self.num_types = num_types

    def forward(self, data):
        one_hot = torch.nn.functional.one_hot(data['atom_types'], num_classes=self.num_types).float()
        data['node_attrs'] = one_hot
        data['node_features'] = one_hot
        data['node_attrs_irreps'] = o3.Irreps(f"{self.num_types}x0e")
        data['node_features_irreps'] = o3.Irreps(f"{self.num_types}x0e")
        return data
    

class SphericalHarmonicEdgeAttrs(nn.Module):
    def __init__(self,
                 l_max: int):
        super().__init__()
        self.l_max = l_max
        self.sh = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(l_max),
                                        normalize=True, normalization='component')
        
    def forward(self, data):
        data['edge_attrs'] = self.sh(data['edge_vectors'])
        data['edge_attrs_irreps'] = o3.Irreps.spherical_harmonics(self.l_max)
        return data
    

class BesselBasis(nn.Module):
    def __init__(self,
                 r_max: float,
                 num_basis: int = 8,
                 trainable: bool = True):
        
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        self.trainable = trainable

        bessel_weights = torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        if trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return (2.0 / self.r_max) * (numerator / x.unsqueeze(-1))


class PolynomialCutoff(nn.Module):
    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.r_max = r_max
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.r_max
        p = self.p

        out = 1.0 - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p)) + \
              (p * (p + 2.0) * torch.pow(x, p + 1.0)) - \
              ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
        
        return out * (x < 1.0)
    

@compile_mode("script")
class RadialBasisEmbedding(nn.Module):
    def __init__(self,
                 basis=BesselBasis,
                 cutoff=PolynomialCutoff,
                 basis_kwargs={},
                 cutoff_kwargs={}):
        
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        
    def forward(self, data):
        x = data['edge_lengths']
        data['edge_embeddings'] = self.basis(x) * self.cutoff(x)[:, None]
        return data      
    