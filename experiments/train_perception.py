"""Train a small CNN ensemble to regress lane error from synthetic road images.

This is the *optional* step that replaces the analytic ``SyntheticPerception``
with a genuine learned vision front-end.  The closed loop, the integrity
monitor and the benchmark all work without it - so build/validate the loop
first, then drop the CNN in.

The ensemble is trained on curvatures inside the ODD (``|kappa| <= kappa_train``)
so that the out-of-distribution scenario is genuinely out of distribution.

Usage::

    python -m experiments.train_perception --members 3 --epochs 8 --n 6000

Requires torch.  Saves ``models/perception.pt``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustdrive.road import RoadRenderer, RoadConditions  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def sample_dataset(n, kappa_train=0.02, seed=0):
    """Render ``n`` (image, [e_y, e_psi]) pairs with in-ODD conditions."""
    rng = np.random.default_rng(seed)
    renderer = RoadRenderer()
    H, W = renderer.cam.H, renderer.cam.W
    X = np.empty((n, 1, H, W), dtype=np.float32)
    Y = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        e_y = rng.uniform(-0.8, 0.8)
        e_psi = rng.uniform(-0.1, 0.1)
        kappa = rng.uniform(-kappa_train, kappa_train)
        cond = RoadConditions(
            brightness=rng.uniform(0.6, 1.0),
            blur=rng.uniform(0.0, 1.2),
            occlusion=rng.uniform(0.0, 0.2),
            noise=rng.uniform(0.01, 0.04),
        )
        X[i, 0] = renderer.render(e_y, e_psi, kappa, cond, rng)
        Y[i] = (e_y, e_psi)
    return X, Y


def train(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from trustdrive.cnn import LaneCNN

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, Y = sample_dataset(args.n, kappa_train=args.kappa_train, seed=123)
    y_mean = Y.mean(0); y_std = Y.std(0) + 1e-6
    Yn = (Y - y_mean) / y_std

    n_val = int(0.15 * len(X))
    Xtr, Ytr = X[:-n_val], Yn[:-n_val]
    Xva, Yva = X[-n_val:], Yn[-n_val:]
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=args.batch, shuffle=True)
    Xva_t = torch.tensor(Xva).to(device); Yva_t = torch.tensor(Yva).to(device)

    state_dicts = []
    for m_idx in range(args.members):
        torch.manual_seed(1000 + m_idx)   # different init per ensemble member
        net = LaneCNN().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        loss_fn = torch.nn.MSELoss()
        for ep in range(args.epochs):
            net.train()
            for xb, yb in tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xva_t), Yva_t).item()
            print(f"  member {m_idx} epoch {ep+1}/{args.epochs}  val_mse={val:.4f}")
        state_dicts.append({k: v.cpu() for k, v in net.state_dict().items()})

    out = os.path.join(MODEL_DIR, "perception.pt")
    torch.save({"state_dicts": state_dicts, "arch": {},
                "target_norm": (y_mean, y_std),
                "kappa_train": args.kappa_train}, out)
    print(f"saved ensemble ({args.members} members) -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Train the CNN perception ensemble")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n", type=int, default=6000, help="dataset size")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kappa-train", type=float, default=0.02, dest="kappa_train")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
