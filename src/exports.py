"""exports.py - writes the trained PINN weights and the physics constants to JSON for MATLAB."""

import os
import json
import numpy as np
import torch

import config as C
from model import CSTR_PINN


def main():
    ck = torch.load(os.path.join(C.OUT_DIR, "pinn_best.pt"), weights_only=False)
    sd = ck["model"]

    def arr(name):
        return sd[name].cpu().numpy().tolist()

    export = {
        "backbone": [
            {"W": arr("backbone.0.weight"), "b": arr("backbone.0.bias")},
            {"W": arr("backbone.2.weight"), "b": arr("backbone.2.bias")},
            {"W": arr("backbone.4.weight"), "b": arr("backbone.4.bias")},
        ],
        "f_head": {"W": arr("f_head.weight"), "b": arr("f_head.bias")},
        "g_head": {"W": arr("g_head.weight"), "b": arr("g_head.bias")},
        "activation": "tanh",
        "n_state": 2, "n_input": 2,

        "xdot_scale": sd["xdot_scale"].cpu().numpy().tolist(),
        "u_scale":    sd["u_scale"].cpu().numpy().tolist(),

        "xscaler_min": np.asarray(ck["xscaler_min"]).tolist(),
        "xscaler_max": np.asarray(ck["xscaler_max"]).tolist(),

        "P":       C.P_MATRIX,
        "steady":  {"CAS": C.CAS, "TS": C.TS, "CA0S": C.CA0S, "QS": C.QS},
        "DU_MAX":  C.DU_MAX,
        "C1_LYAP": C.C1_LYAP,
        "RHO_S":   C.RHO_S,
        "DT":      C.DT,
        "ROLLOUT_DT": C.ROLLOUT_DT,
    }

    out = os.path.join(C.OUT_DIR, "pinn_weights.json")
    with open(out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Exported -> {out}")

    # I also save a reference forward pass so MATLAB can check itself.
    model = CSTR_PINN()
    model.load_state_dict(sd)
    model.eval()
    x_test = torch.tensor([[C.CAS, C.TS], [1.0, 450.0], [3.0, 380.0]], dtype=torch.float32)
    xmin = torch.tensor(ck["xscaler_min"]); xmax = torch.tensor(ck["xscaler_max"])
    with torch.no_grad():
        xs = (x_test - xmin) / (xmax - xmin)
        f, g = model.f_g(xs)

    print("\nReference forward pass (verify MATLAB reproduces these):")
    for i in range(len(x_test)):
        print(f"  x = [{x_test[i,0]:.4f}, {x_test[i,1]:.4f}]")
        print(f"    f = [{f[i,0]:.6e}, {f[i,1]:.6e}]")
        print(f"    g = [[{g[i,0,0]:.6e}, {g[i,0,1]:.6e}], "
              f"[{g[i,1,0]:.6e}, {g[i,1,1]:.6e}]]")

    ref = {"x": x_test.numpy().tolist(),
           "f": f.numpy().tolist(), "g": g.numpy().tolist()}
    with open(os.path.join(C.OUT_DIR, "pinn_reference.json"), "w") as fp:
        json.dump(ref, fp, indent=2)
    print("\nSaved reference -> outputs/pinn_reference.json")


if __name__ == "__main__":
    main()