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
