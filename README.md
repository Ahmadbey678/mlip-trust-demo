# MLIP trust demo: when can you trust a machine-learned potential?

A small learning project. It trains a neural-network potential on bulk copper, then tests it on structures it has not seen (vacancy, surface slab, small cluster, compressed bulk) and checks whether the disagreement between ensemble members flags the bad predictions.

Live page: https://ahmadbey678.github.io/mlip-trust-demo/

![Error by structure type](fig_error_by_structure.png)

![Ensemble uncertainty versus error](fig_uncertainty_vs_error.png)

## What it does

- Ground truth: ASE `EMT` potential for Cu (a cheap stand-in for DFT, not DFT)
- Training data: 300 bulk cells with random lattice constants and random atomic displacements
- Descriptors: Behler-Parrinello-style radial and angular symmetry functions
- Model: ensemble of 8 small MLPs (scikit-learn, L-BFGS) predicting energy per atom
- Uncertainty: standard deviation across the 8 models
- Results (seeded, reproducible): bulk MAE about 1.5 meV/atom; cluster about 389 meV/atom (roughly 260 times higher); Spearman correlation between ensemble std and error 0.91 over 150 test structures

## Run

```
pip install -r requirements.txt
python mlip_demo.py        # about 20 s, writes results.json
python make_figures.py    # writes the two PNGs
# open index.html in a browser to see the project page
```

## Files

- `mlip_demo.py`: data generation, descriptors, training, evaluation
- `make_figures.py`: draws the three figures (error by structure, uncertainty vs error, parity plots) from `results.json`
- `results.json`: saved results from the seeded run (the page shows the same numbers)
- `index.html`: the project page (served by GitHub Pages)

## Limits

EMT is not DFT; one element only; energy per atom from descriptors averaged over atoms (not a sum of per-atom energies); no forces; small test sets. Small test-set errors are sensitive to the random initialization (removing 20 unused zero columns from the descriptor array moved the vacancy error from 4.9 to 11.0 meV/atom and the slab error from 19.5 to 9.7); bulk stayed lowest and cluster/compressed bulk highest. My first run used the Adam solver and converged poorly, so I switched to L-BFGS. The ensemble std is a warning signal, not a calibrated error bar (it overestimates the error on the surface slab).

## License

MIT. See `LICENSE`.
