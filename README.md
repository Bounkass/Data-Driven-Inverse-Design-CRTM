# Data-Driven Inverse Design of the Injection Concentration in CRTM

A surrogate **inverse model** that finds the resin injection concentration $C_0(t)$ needed to obtain a prescribed particle distribution $Q(x)$ in a part made by **Compression Resin Transfer Molding (CRTM)**.

Gaussian-process training data + PCA + MLP.

## Context

In CRTM, the resin is injected into a gap above a dry fibrous preform, and the mould is then closed. The compression drives the resin through the thickness of the reinforcement. When the resin carries particles, part of them is retained by the fibres (filtration). The final particle distribution $Q(x)$ in the part therefore depends on how the injected concentration $C_0(t)$ varies over time.

Controlling $C_0(t)$ makes it possible to produce **functionally graded (FG) materials**.

**Inverse problem.** Given a target distribution $Q^\star(x)$, find $C_0(t)$ such that the process yields $Q(x) \approx Q^\star(x)$.

**Approach.** Instead of coupling a forward solver with an optimisation loop, the inverse map is learned directly:

$$\mathcal{G}_\theta : Q(x) \longmapsto C_0(t)$$

Once trained, the model returns $C_0$ for a new target in a single forward pass.

## Method

| Step | Details |
|---|---|
| Data | 1000 CRTM simulations. $C_0(t)$ is drawn from Gaussian processes (242 time steps); $Q(x)$ is on 301 grid points |
| Split | 90 % train / 10 % test |
| Preprocessing | Standardisation, then PCA keeping 99.999 % of variance: 15 modes for $Q$, 14 modes for $C_0$ |
| Network | MLP 256 → 128 → 64, swish activation, L2 regularisation $10^{-4}$ |
| Training | Huber loss, Adam with cosine learning-rate decay, early stopping (patience 400), up to 3000 epochs |
| Validation | Predicted $C_0$ profiles are re-simulated with the CRTM solver and compared with the targets |

## Target profiles

Six target distributions on $x \in [0,1]$: uniform (**UD**), and functionally graded **FG-O**, **FG-X**, **FG-XO**, **FG-OX** and **FG-NI** (non-smooth, piecewise linear). Their definitions are given in the notebook.

## Results

- **Test set** ($C_0$): RMSE = 0.00151, MAE = 0.00078, $R^2$ = 0.99973.
- **Re-simulation**: the distributions obtained with the predicted $C_0$ match the six targets with a relative L2 error of **0.3 % to 1.5 %**.
- **Compared with the dataset**: the closest existing simulation reaches only 5–12 % relative error. The network is better by a factor of **6.5 to 25**, so it builds new injection profiles rather than retrieving an existing one.
- **Interpolation regime**: in the PCA space of $Q$, all targets lie within the training distribution.

The remaining error is concentrated at the boundaries $x = 0$ and $x = 1$ and at the slope discontinuities of FG-NI.

## Repository structure

```
.
├── inverse_design_CRTM.ipynb      # main notebook
├── requirements.txt
├── data/
│   ├── ConcentrationIGP0.txt             # C_0(t), shape (1000, 242)
│   ├── quantite_particules_donneesGP0.txt # Q(x),   shape (1000, 301)
│   └── resimulated/                      # Q re-simulated from the predicted C_0
├── results/predicted_C0/          # C_0 predicted for each target (created by the notebook)
├── models/                        # best trained model (created by the notebook)
└── figures/                       # saved figures (created by the notebook)
```

## How to run

```bash
git clone https://github.com/<your-username>/Data-Driven-Inverse-Design-CRTM.git
cd Data-Driven-Inverse-Design-CRTM
pip install -r requirements.txt
jupyter notebook inverse_design_CRTM.ipynb
```

The data files must be placed in `data/` as shown above. Section 8 (validation by re-simulation) also needs the files in `data/resimulated/`, which are produced by running the CRTM solver (not included) on the profiles in `results/predicted_C0/`.
