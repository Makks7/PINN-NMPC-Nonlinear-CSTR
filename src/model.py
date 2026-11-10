"""model.py - the PINN in control-affine form, xdot = f_nn(x) + g_nn(x) @ u."""

import torch
import torch.nn as nn
import config as C


class CSTR_PINN(nn.Module):
    def __init__(self, n_state=2, n_input=2, hidden=None):
        super().__init__()
        self.n_state = n_state
        self.n_input = n_input
        hidden = hidden or C.HIDDEN_LAYERS
        act = nn.Tanh() if C.ACTIVATION == "tanh" else nn.ReLU()

        # I share one backbone that reads the state, then split into two heads.
        layers, prev = [], n_state
        for h in hidden:
            layers += [nn.Linear(prev, h), act]
            prev = h
        self.backbone = nn.Sequential(*layers)

        # I need f and g separately, the Sontag law can't be built from a lumped xdot.
        self.f_head = nn.Linear(prev, n_state)
        self.g_head = nn.Linear(prev, n_state * n_input)

        # I keep the scales as buffers so they save and move with the model.
        self.register_buffer("xdot_scale", torch.tensor(C.XDOT_SCALE))
        self.register_buffer("u_scale",    torch.tensor(C.U_SCALE))

    def f_g(self, x_scaled):
        """I return f and g in physical units, un-scaling the dimensionless heads."""
        h     = self.backbone(x_scaled)
        f_hat = self.f_head(h)
        g_hat = self.g_head(h).view(-1, self.n_state, self.n_input)

        f = f_hat * self.xdot_scale
        g = g_hat * self.xdot_scale.view(1, -1, 1) / self.u_scale.view(1, 1, -1)
        return f, g

    def forward(self, x_scaled, u_phys):
        f, g = self.f_g(x_scaled)
        gu   = torch.bmm(g, u_phys.unsqueeze(-1)).squeeze(-1)
        return f + gu


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CSTR_PINN()
    print(m)
    print("Trainable params:", count_params(m))
    f, g = m.f_g(torch.rand(8, 2))
    print("f:", f.shape, " g:", g.shape, " xdot:", m(torch.rand(8, 2), torch.randn(8, 2)).shape)