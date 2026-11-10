"""train.py - trains the PINN (three-part loss, Adam, LR decay, early stopping)."""

import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as C
from preprocess import make_datasets
from model import CSTR_PINN, count_params
from losses import total_loss


def make_loader(split, batch_size, shuffle):
    """I batch over row indices and gather from the split dict inside the loop."""
    n = split["X_scaled"].shape[0]
    return DataLoader(TensorDataset(torch.arange(n)), batch_size=batch_size, shuffle=shuffle)


def gather(split, idx, device):
    return {
        "X_scaled": split["X_scaled"][idx].to(device),
        "Y_scaled": split["Y_scaled"][idx].to(device),
        "X_raw":    split["X_raw"][idx].to(device),
        "Y_raw":    split["Y_raw"][idx].to(device),
    }


def evaluate(model, split, device):
    model.eval()
    with torch.no_grad():
        b = gather(split, torch.arange(split["X_scaled"].shape[0]), device)
        _, comp = total_loss(model, b, device)
    return comp


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = make_datasets()
    print("Samples:", data["n"])
    train, val = data["train"], data["val"]

    model = CSTR_PINN().to(device)
    print(f"Model params: {count_params(model)}")

    opt   = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=C.LR_DECAY_STEP,
                                            gamma=C.LR_DECAY_GAMMA)

    loader   = make_loader(train, C.BATCH_SIZE, shuffle=True)
    best_val = float("inf")
    patience = 0
    history  = []

    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        for (batch_idx,) in loader:
            b = gather(train, batch_idx, device)
            loss, _ = total_loss(model, b, device)
            opt.zero_grad()
            loss.backward()
            # I clip gradients: the Sontag term can produce huge spikes that wreck training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()

        tr = evaluate(model, train, device)
        va = evaluate(model, val, device)
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}] train {tr['total']:.5f} "
                  f"(d {tr['data']:.4f} p {tr['phys']:.4f} g {tr['g']:.4f} L {tr['lyap']:.4f}) | "
                  f"val {va['total']:.5f}")

        # I keep the best model on validation and stop when it plateaus.
        if va["total"] < best_val - 1e-6:
            best_val = va["total"]
            patience = 0
            torch.save({"model": model.state_dict(),
                        "xscaler_min": data["xscaler"].min_,
                        "xscaler_max": data["xscaler"].max_,
                        "yscaler_min": data["yscaler"].min_,
                        "yscaler_max": data["yscaler"].max_,
                        "config": {k: getattr(C, k) for k in
                                   ["HIDDEN_LAYERS", "DT", "P_MATRIX",
                                    "XDOT_SCALE", "U_SCALE", "U_LO", "U_HI"]}},
                       os.path.join(C.OUT_DIR, "pinn_best.pt"))
        else:
            patience += 1
            if patience >= C.PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.5f})")
                break

    with open(os.path.join(C.OUT_DIR, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    print("Saved best model -> outputs/pinn_best.pt")
    print("Saved history    -> outputs/history.csv")


if __name__ == "__main__":
    main()