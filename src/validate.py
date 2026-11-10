"""validate.py - validation in three parts: f/g recovery, V_dot over the region, and a closed-loop rollout."""

import os
import numpy as np
import torch

import config as C
import physics as phys
from model import CSTR_PINN


def load_model(device):
    ck = torch.load(os.path.join(C.OUT_DIR, "pinn_best.pt"),
                    map_location=device, weights_only=False)
    m = CSTR_PINN().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    xmin = torch.tensor(ck["xscaler_min"]).to(device)
    xmax = torch.tensor(ck["xscaler_max"]).to(device)
    return m, xmin, xmax


def scale_x(x_abs, xmin, xmax):
    return (x_abs - xmin) / (xmax - xmin)


def true_f_g(x_abs):
    """I get the analytic f(x) and g(x) so I can check Lemma 3.1 directly."""
    CA, T = x_abs[:, 0], x_abs[:, 1]
    rate = C.K0 * torch.exp(-C.E / (C.R * T)) * CA**2

    # f(x): the drift term (u = 0)
    f1 = (C.F / C.V) * (0.0 - CA) - rate
    f2 = (C.F / C.V) * (C.T0 - T) + (-C.DH / (C.RHOL * C.CP)) * rate
    f = torch.stack([f1, f2], dim=1)

    # g(x): constant here: dCA/dCA0 = F/V, dT/dQ = 1/(rho*Cp*V)
    n = x_abs.shape[0]
    g = torch.zeros(n, 2, 2, dtype=x_abs.dtype, device=x_abs.device)
    g[:, 0, 0] = C.F / C.V
    g[:, 1, 1] = 1.0 / (C.RHOL * C.CP * C.V)
    return f, g


def r2(t, p):
    """I return nan when the target is constant, R2 is undefined there (true g is constant)."""
    var = ((t - t.mean()) ** 2).sum()
    if var < 1e-12:
        return float("nan")
    return 1 - ((t - p) ** 2).sum() / var


def grid(n=113):   # 113^2 = 12,769 ~ the paper's 12,603 states
    ca = np.linspace(-2.0, 2.0, n)
    t = np.linspace(-100.0, 100.0, n)
    CA, T = np.meshgrid(ca, t)
    x_dev = np.stack([CA.ravel(), T.ravel()], axis=1).astype(np.float32)
    x_abs = x_dev + np.array([C.CAS, C.TS], dtype=np.float32)
    return CA, T, torch.tensor(x_abs)


def rollout(model, xmin, xmax, x0, steps, dt, device):
    """I simulate the TRUE plant under the PINN's Sontag controller."""
    x = torch.tensor(x0, dtype=torch.float32, device=device).view(1, 2)
    traj, us = [x.cpu().numpy().ravel()], []
    for _ in range(steps):
        with torch.no_grad():
            f, g = model.f_g(scale_x(x, xmin, xmax))
            u = torch.tensor([[C.CA0S, C.QS]], device=x.device) \
                + phys.sontag_controller(f, g, x)
        us.append(u.cpu().numpy().ravel())
        # I advance the real plant (RK4), not the model, this is the real test.
        k1 = phys.cstr_ode(x, u)
        k2 = phys.cstr_ode(x + .5 * dt * k1, u)
        k3 = phys.cstr_ode(x + .5 * dt * k2, u)
        k4 = phys.cstr_ode(x + dt * k3, u)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.cpu().numpy().ravel())
        if not np.all(np.isfinite(traj[-1])):
            break
    return np.array(traj), np.array(us)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, xmin, xmax = load_model(device)

    CA, T, x_abs = grid()
    x_abs = x_abs.to(device)

    with torch.no_grad():
        xs = scale_x(x_abs, xmin, xmax)
        f_nn, g_nn = model.f_g(xs)
        f_t, g_t = true_f_g(x_abs)

    # 1. Lemma 3.1: did f_nn and g_nn separately recover the true f and g?
    print("=" * 66)
    print("1. f/g RECOVERY over the operating region  (paper Lemma 3.1)")
    print("=" * 66)
    fn, ft = f_nn.cpu().numpy(), f_t.cpu().numpy()
    print(f"{'Component':<16} {'R2':>8} {'RMSE':>12}")
    for i, nm in enumerate(["f1 (dCA/dt)", "f2 (dT/dt)"]):
        print(f"{nm:<16} {r2(ft[:,i], fn[:,i]):>8.4f} "
              f"{np.sqrt(((ft[:,i]-fn[:,i])**2).mean()):>12.4g}")
    gn, gt = g_nn.cpu().numpy(), g_t.cpu().numpy()
    # I report g as % error against its true constant value, R2 is undefined for a constant.
    for (i, j), nm, tv in [((0, 0), "g11 (dCA/dCA0)", C.F / C.V),
                           ((1, 1), "g22 (dT/dQ)", 1.0 / (C.RHOL * C.CP * C.V))]:
        rmse = np.sqrt(((gt[:, i, j] - gn[:, i, j]) ** 2).mean())
        print(f"{nm:<16} {'(const)':>8} {rmse:>12.4g}   true={tv:<10.5g} err={rmse/tv*100:5.2f}%")

    # 2. V_dot over the region (paper Fig. 5)
    with torch.no_grad():
        us = torch.tensor([C.CA0S, C.QS], device=device)
        u = us + phys.sontag_controller(f_nn, g_nn, x_abs)
        gu = torch.bmm(g_nn, u.unsqueeze(-1)).squeeze(-1)
        Vdot = phys.v_dot(x_abs, f_nn + gu).cpu().numpy()
        V = phys.lyapunov_V(x_abs).cpu().numpy()

    rho = 368.0
    inside = V <= rho
    n_in = int(inside.sum())
    n_pos = int((Vdot[inside] >= 0).sum())
    print("\n" + "=" * 66)
    print("2. LYAPUNOV STABILITY over the region  (paper Fig. 5)")
    print("=" * 66)
    print(f"States inside ellipse (V <= {rho}): {n_in}")
    print(f"  V_dot <  0 : {n_in-n_pos}  ({(n_in-n_pos)/n_in*100:.2f}%)")
    print(f"  V_dot >= 0 : {n_pos}  ({n_pos/n_in*100:.2f}%)")
    print(f"  V_dot range: [{Vdot[inside].min():.1f}, {Vdot[inside].max():.1f}]")
    print("  (paper: 178/12603 positive -> 98.59% negative)")

    # 3. Closed-loop rollout from four initial states (paper Fig. 8)
    print("\n" + "=" * 66)
    print("3. CLOSED-LOOP ROLLOUT on the TRUE plant  (paper Fig. 8)")
    print("=" * 66)
    x0s = [[C.CAS - 1.0, C.TS + 60.0], [C.CAS + 1.0, C.TS - 60.0],
           [C.CAS - 1.4, C.TS + 30.0], [C.CAS + 1.4, C.TS - 30.0]]
    # I update the controller every 0.001 hr, sample-and-hold at 0.01 hr
    # chatters and fails (the Sontag gain near the origin is huge).
    steps, dt = 3000, C.ROLLOUT_DT
    trajs = []
    print(f"{'x0 (dev)':<22} {'final V':>12} {'|x-xs| final':>14} {'converged':>11}")
    for x0 in x0s:
        tr, us = rollout(model, xmin, xmax, x0, steps, dt, device)
        trajs.append((np.array(x0), tr, us))
        xf = torch.tensor(tr[-1:], dtype=torch.float32)
        Vf = phys.lyapunov_V(xf).item()
        err = np.linalg.norm(tr[-1] - np.array([C.CAS, C.TS]))
        ok = "YES" if Vf < C.RHO_S else "no"
        d0 = np.array(x0) - np.array([C.CAS, C.TS])
        print(f"[{d0[0]:+.2f}, {d0[1]:+.1f}]{'':<8} {Vf:>12.3f} {err:>14.4f} {ok:>11}")

    np.savez(os.path.join(C.OUT_DIR, "validate_data.npz"),
             CA=CA, T=T, Vdot=Vdot, V=V, rho=rho,
             f_nn=fn, f_true=ft, g_nn=gn, g_true=gt,
             trajs=np.array([t[1] for t in trajs], dtype=object), allow_pickle=True)
    print("\nSaved outputs/validate_data.npz (plots read from this)")


if __name__ == "__main__":
    main()