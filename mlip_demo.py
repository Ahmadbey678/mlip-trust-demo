"""Machine-learned interatomic potential (MLIP) demo: when can you trust it?

Ground truth: ASE's EMT potential for copper (a cheap stand-in for DFT).
Model: Behler-Parrinello-style symmetry-function descriptors -> small neural
network ensemble that predicts energy per atom.

Question: trained only on bulk copper, how does the model do on
  (1) more bulk data it has not seen (in-distribution),
  (2) a surface slab, a vacancy, and a small cluster (out-of-distribution)?
And does the ensemble disagreement flag the cases where it is wrong?
"""
import json
import numpy as np
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)
RC = 5.0  # cutoff, angstrom
ETAS = [0.02, 0.06, 0.15, 0.4, 1.0]  # radial widths
RS = [2.0, 2.6, 3.2, 3.8, 4.4]  # radial centers
ANG = [(0.01, 1, 1), (0.01, 1, -1), (0.03, 1, 1), (0.03, 1, -1), (0.08, 2, 1), (0.08, 2, -1)]


def cutoff(r):
    return 0.5 * (np.cos(np.pi * r / RC) + 1.0) * (r < RC)


def descriptors(atoms):
    """Per-atom symmetry functions (radial G2 and angular G4), averaged over atoms."""
    n = len(atoms)
    # neighbours within cutoff, with periodic images
    from ase.neighborlist import neighbor_list
    i, j, d, D = neighbor_list("ijdD", atoms, RC)
    nrad = len(ETAS)  # 5 radial functions (eta, Rs pairs), then len(ANG) angular ones
    feats = np.zeros((n, nrad + len(ANG)))
    for a in range(n):
        m = i == a
        da, Da = d[m], D[m]
        fc = cutoff(da)
        for k, (eta, rs) in enumerate(zip(ETAS, RS)):
            feats[a, k] = np.sum(np.exp(-eta * (da - rs) ** 2) * fc)
        # angular terms over neighbour pairs
        if len(da) > 1:
            cosm = (Da @ Da.T) / np.outer(da, da)
            iu = np.triu_indices(len(da), 1)
            c = cosm[iu]
            r1, r2 = da[iu[0]], da[iu[1]]
            r12 = np.linalg.norm(Da[iu[0]] - Da[iu[1]], axis=1)
            f = cutoff(r1) * cutoff(r2) * cutoff(r12)
            for k, (eta, zeta, lam) in enumerate(ANG):
                t = (1 + lam * c).clip(0) ** zeta * np.exp(-eta * (r1**2 + r2**2 + r12**2)) * f
                feats[a, nrad + k] = 2 ** (1 - zeta) * np.sum(t)
    return feats.mean(axis=0)


def energy_per_atom(atoms):
    atoms.calc = EMT()
    return atoms.get_potential_energy() / len(atoms)


def make_bulk(n, rattle, a_range):
    out = []
    for _ in range(n):
        a = RNG.uniform(*a_range)
        at = bulk("Cu", "fcc", a=a, cubic=True).repeat((2, 2, 2))
        at.rattle(stdev=RNG.uniform(*rattle), seed=int(RNG.integers(1e9)))
        out.append(at)
    return out


def make_slabs(n):
    out = []
    for _ in range(n):
        at = fcc111("Cu", size=(3, 3, 4), vacuum=6.0, a=RNG.uniform(3.58, 3.64))
        at.pbc = True
        at.rattle(stdev=RNG.uniform(0.0, 0.05), seed=int(RNG.integers(1e9)))
        out.append(at)
    return out


def make_vacancies(n):
    out = []
    for _ in range(n):
        at = bulk("Cu", "fcc", a=RNG.uniform(3.58, 3.64), cubic=True).repeat((3, 3, 3))
        del at[int(RNG.integers(len(at)))]
        at.rattle(stdev=RNG.uniform(0.0, 0.05), seed=int(RNG.integers(1e9)))
        out.append(at)
    return out


def make_clusters(n):
    from ase.cluster import Icosahedron
    out = []
    for _ in range(n):
        at = Icosahedron("Cu", noshells=2, latticeconstant=RNG.uniform(3.58, 3.64))
        at.center(vacuum=6.0)
        at.pbc = True
        at.rattle(stdev=RNG.uniform(0.0, 0.05), seed=int(RNG.integers(1e9)))
        out.append(at)
    return out


def featurize(structs):
    X = np.array([descriptors(s) for s in structs])
    y = np.array([energy_per_atom(s) for s in structs])
    return X, y


def main():
    train = make_bulk(300, (0.0, 0.10), (3.45, 3.75))
    test_bulk = make_bulk(60, (0.0, 0.10), (3.45, 3.75))
    sets = {
        "bulk (seen type)": test_bulk,
        "vacancy": make_vacancies(20),
        "surface slab": make_slabs(20),
        "small cluster": make_clusters(20),
        "bulk, strongly compressed": make_bulk(30, (0.0, 0.05), (3.05, 3.30)),
    }
    Xtr, ytr = featurize(train)
    sc = StandardScaler().fit(Xtr)
    models = []
    for seed in range(8):
        m = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs", max_iter=3000,
                         tol=1e-10, random_state=seed, alpha=1e-4)
        m.fit(sc.transform(Xtr), ytr)
        models.append(m)

    results = {}
    for name, structs in sets.items():
        X, y = featurize(structs)
        preds = np.array([m.predict(sc.transform(X)) for m in models])
        mean, std = preds.mean(0), preds.std(0)
        err = np.abs(mean - y)
        results[name] = {
            "n": len(structs),
            "mae_meV_per_atom": float(1000 * err.mean()),
            "mean_ensemble_std_meV": float(1000 * std.mean()),
            "err": (1000 * err).tolist(),
            "std": (1000 * std).tolist(),
            "true": y.tolist(),
            "pred": mean.tolist(),
        }
        print(f"{name:28s} MAE {1000*err.mean():9.2f} meV/atom   ensemble std {1000*std.mean():8.2f} meV/atom")

    # does uncertainty rank errors? Spearman correlation over all test points
    from scipy.stats import spearmanr
    all_err = np.concatenate([r["err"] for r in results.values()])
    all_std = np.concatenate([r["std"] for r in results.values()])
    rho = float(spearmanr(all_std, all_err).statistic)
    print("Spearman(ensemble std, abs error) over all test points:", round(rho, 3))
    results["_spearman"] = rho
    results["_train_n"] = len(train)
    json.dump(results, open("results.json", "w"))


if __name__ == "__main__":
    main()
