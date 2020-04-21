import torch 
import torch.nn as nn 

class METLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.poly_degree = config.met_poly_degree
        hidden_size = config.met_poly_degree + 1 # + config.feature_size
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) 
                                      for _ in range(config.met_layers)])
        self.last = nn.Linear(hidden_size, 1)

    def forward(self, x, score, mask):
        # all inputs are of shape [B, P]

        pt, phi = x[:,:,0], x[:,:,2]

        basis = torch.stack([score ** p for p in range(self.poly_degree+1)], dim=-1) # [B,P,D]
        # basis = torch.cat([basis, x], dim=-1) 
        for l in self.linears:
            basis = l(basis)
            basis = torch.tanh(basis)

        weight = self.last(basis).squeeze(-1)  # [B,P]

        pt = pt * weight
        
        # [B]
        met_x = torch.sum(pt * torch.sin(phi) * mask, dim=-1)
        met_y = torch.sum(pt * torch.cos(phi) * mask, dim=-1)

        met = torch.sqrt(met_x ** 2 + met_y ** 2)

        return met, weight
