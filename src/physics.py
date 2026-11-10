"""physics.py - the CSTR model equations, the Lyapunov function and the Sontag control law."""

import torch
import config as C


def cstr_ode(x, u):
    """I compute the true plant derivative xdot = f(x) + g(x)u."""
    CA, T   = x[:, 0], x[:, 1]
    CA0, Q  = u[:, 0], u[:, 1]
    rate = C.K0 * torch.exp(-C.E / (C.R * T)) * CA**2
    dCA  = (C.F / C.V) * (CA0 - CA) - rate
    dT   = (C.F / C.V) * (C.T0 - T) + (-C.DH / (C.RHOL * C.CP)) * rate \
           + Q / (C.RHOL * C.CP * C.V)
    return torch.stack([dCA, dT], dim=1)


_P = torch.tensor(C.P_MATRIX, dtype=torch.float32)


def lyapunov_V(x):
    """I evaluate V = z'Pz in deviation coords z = x - x_steady."""
    z = x - torch.tensor([C.CAS, C.TS], dtype=x.dtype, device=x.device)
    P = _P.to(x.device, x.dtype)
    return torch.einsum("bi,ij,bj->b", z, P, z)


def lyapunov_grad(x):
    """I get dV/dx = 2Pz, needed for the chain rule in V_dot."""
    z = x - torch.tensor([C.CAS, C.TS], dtype=x.dtype, device=x.device)
    P = _P.to(x.device, x.dtype)
    return 2.0 * z @ P.T


def v_dot(x, xdot):
    """I form V_dot = dV/dx . xdot."""
    return (lyapunov_grad(x) * xdot).sum(dim=1)


def sontag_controller(f_nn, g_nn, x_phys):
    """I compute the Sontag law (Eq. 3a/3b) in DEVIATION-input form.

    The paper works in deviation inputs du = u - us, so I build the drift at
    the steady-state input, f_bar = f + g @ us, run Sontag on (f_bar, g), and
    clamp du to the DEVIATION box +/-DU_MAX. I return du; callers add us.
    """
    us = torch.tensor([C.CA0S, C.QS], dtype=f_nn.dtype, device=f_nn.device)
    f_bar = f_nn + torch.bmm(g_nn, us.expand(f_nn.shape[0], 2).unsqueeze(-1)).squeeze(-1)

    gradV = lyapunov_grad(x_phys)
    LfV   = (gradV * f_bar).sum(dim=1)

    # I run Sontag in NORMALISED input coords (u_hat = du/DU_MAX, |u_hat|<=1).
    # Raw units make the law dump all effort into CA0 and ignore Q entirely
    # the formula is not scale-invariant, so I balance the channels first.
    du_max = torch.tensor(C.DU_MAX, dtype=f_nn.dtype, device=f_nn.device)
    g_hat  = g_nn * du_max.view(1, 1, -1)
    LgV    = torch.bmm(gradV.unsqueeze(1), g_hat).squeeze(1)

    # I floor ||LgV||^2 hard, a near-singular g otherwise makes 1/||LgV||^2 explode.
    LgV_sq = (LgV**2).sum(dim=1).clamp(min=1e-3)
    inner  = LfV**2 + LgV_sq**2
    coef   = -(LfV + torch.sqrt(inner + 1e-8)) / LgV_sq
    u_hat  = torch.clamp(coef.unsqueeze(1) * LgV, min=-1.0, max=1.0)

    return u_hat * du_max