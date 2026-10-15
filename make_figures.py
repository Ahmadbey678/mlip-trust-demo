import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results.json"))
names = [k for k in R if not k.startswith("_")]
mae = [R[k]["mae_meV_per_atom"] for k in names]
std = [R[k]["mean_ensemble_std_meV"] for k in names]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(names)); w = 0.38
ax.bar(x - w/2, mae, w, label="Actual error (MAE)", color="#c0392b")
ax.bar(x + w/2, std, w, label="Model's own uncertainty (ensemble std)", color="#2c6fbb")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([n.replace(", ", ",\n").replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylabel("meV per atom (log scale)"); ax.legend(fontsize=8)
ax.set_title("Trained on bulk copper only: error grows on unseen structures")
plt.tight_layout(); plt.savefig("fig_error_by_structure.png", dpi=160)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for k, c in zip(names, ["#2c6fbb", "#e69f00", "#c0392b", "#7b3fa0", "#555555"]):
    ax.scatter(R[k]["std"], R[k]["err"], s=14, label=k, color=c, alpha=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Ensemble std (meV/atom)"); ax.set_ylabel("Absolute error (meV/atom)")
ax.set_title(f"Uncertainty vs error (Spearman = {R['_spearman']:.2f})", fontsize=10)
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("fig_uncertainty_vs_error.png", dpi=160)

# parity plot: predicted vs reference energy per atom, one panel per test set
fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
for ax, k in zip(axes, names):
    t = np.array(R[k]["true"]); p = np.array(R[k]["pred"])
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    pad = 0.05 * (hi - lo) + 1e-3
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888", lw=1, ls="--", zorder=1)
    ax.scatter(t, p, s=12, color="#2f7d70", alpha=0.85, zorder=2)
    ax.set_title(k, fontsize=9)
    ax.set_xlabel("Reference (eV/atom)", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Model prediction (eV/atom)", fontsize=8)
plt.tight_layout(); plt.savefig("fig_parity.png", dpi=160)
