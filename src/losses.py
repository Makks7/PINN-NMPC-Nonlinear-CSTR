"""losses.py - the training loss: data MSE + physics residual + the Lyapunov stability term."""

import torch
import torch.nn.functional as Fnn
import physics as phys
import config as C


def data_loss(xdot_pred, xdot_true, scale):
    """I normalise per channel so dT/dt doesn't swamp dCA/dt."""
    return (((xdot_pred - xdot_true) / scale)**2).mean()


def physics_loss(xdot_pred, x_phys, u_phys, scale):
    """I penalise the residual against the known CSTR ODE at the same (x,u)."""
    xdot_true = phys.cstr_ode(x_phys, u_phys)
    return (((xdot_pred - xdot_true) / scale)**2).mean()


def g_loss(model, x_scaled):
    """I supervise g_nn directly against the known analytic g(x).

    The sum f+g@u fits the data perfectly while the SPLIT stays wrong, and the
    Sontag law is built from the split, so I pin g to the physics.
    """
    _, g_nn = model.f_g(x_scaled)
    n = g_nn.shape[0]
    g_true = torch.zeros_like(g_nn)
    g_true[:, 0, 0] = C.F / C.V
    g_true[:, 1, 1] = 1.0 / (C.RHOL * C.CP * C.V)
    # I normalise per entry by the true scale so both channels count equally.
    gs = torch.tensor([[C.F / C.V, 1.0], [1.0, 1.0 / (C.RHOL * C.CP * C.V)]],
                      dtype=g_nn.dtype, device=g_nn.device)
    return (((g_nn - g_true) / gs)**2).mean()


def lyapunov_loss(model, x_scaled, x_phys):
    """I evaluate V_dot under the network's OWN Sontag controller, not the dataset's u.

    That's the faithful form of Eq. 10b: I'm asking whether the controller this
    model implies would actually drive V down.
    """
    f_nn, g_nn = model.f_g(x_scaled)

    us = torch.tensor([C.CA0S, C.QS], dtype=f_nn.dtype, device=f_nn.device)
    du       = phys.sontag_controller(f_nn, g_nn, x_phys)
    u_abs    = us + du
    gu       = torch.bmm(g_nn, u_abs.unsqueeze(-1)).squeeze(-1)
    xdot_cl  = f_nn + gu

    Vdot = phys.v_dot(x_phys, xdot_cl)
    V    = phys.lyapunov_V(x_phys)

    # I only penalise states outside the small level set rho_s.
    outside = (V > C.RHO_S).float()
    penalty = Fnn.relu(Vdot + C.C1_LYAP)
    return (outside * penalty).mean()


def total_loss(model, batch, device):
    X_scaled = batch["X_scaled"].to(device)
    X_raw    = batch["X_raw"].to(device)
    Y_raw    = batch["Y_raw"].to(device)

    x_phys = X_raw[:, 0:2]
    u_phys = X_raw[:, 2:4]

    # I detach the RMS scale so it acts as a constant, keeping both terms O(1).
    scale = torch.sqrt((Y_raw**2).mean(dim=0) + 1e-8).detach()

    xdot_pred = model(X_scaled, u_phys)

    Ld = data_loss(xdot_pred, Y_raw, scale)
    Lp = physics_loss(xdot_pred, x_phys, u_phys, scale)
    Lg = g_loss(model, X_scaled)
    Ll = lyapunov_loss(model, X_scaled, x_phys)

    total = Ld + C.LAMBDA_PHYS * Lp + C.LAMBDA_G * Lg + C.LAMBDA_LYAP * Ll
    return total, {"data": Ld.item(), "phys": Lp.item(), "g": Lg.item(),
                   "lyap": Ll.item(), "total": total.item()}