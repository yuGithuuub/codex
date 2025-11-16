# Project: Microenvironment-driven two-parameter model for metastatic lesion dynamics

## Data

- Main data file: `data/H22_VOLUME_LS_v2.csv`
- Each row is a **single metastatic lesion** reconstructed from whole-lung lightsheet imaging.
- Columns:
  - `MouseID`           : mouse ID
  - `TimePoint (Day)`   : categorical time point, values: `D1`, `D4`, `D7`, `D14`
  - `LesionID`          : lesion ID within mouse
  - `Volume_um3`        : lesion volume in µm³  (if the actual column name is `Volume_um³`, adapt in code)
  - `logvolume`         : log-transformed volume

- Detection / reconstruction lower bound:
  - `v_min = 10` µm³
  - All model fitting must be performed on `Volume_um3 >= v_min`.

## Modelling goals

We want to implement the **Section 3 (Model Fits)** of the Supplementary Note in code:

1. For each time point (`D1`, `D4`, `D7`, `D14`), fit:
   - a **truncated single-exponential model** for lesion volume distribution
   - a **truncated bi-exponential mixture model** motivated by the two-state microenvironment model.

2. Use **maximum likelihood estimation (MLE)** with:
   - truncation at `v_min = 10 µm³`
   - **multi-start optimisation** and **BFGS** for the mixture model
   - **reparameterisation** for numerical stability:
     - `a2 = logit^{-1}(w)`   → mixing weight of the “slow growth / escape” component
     - `a1 = 1 - a2`
     - `v0 = softplus(u)`     → scale of the “fast clearance / small lesion” component
     - `v1 = v0 + softplus(z)` → scale of the “escape / large lesion” component

3. For each model × time point, compute and export:
   - log-likelihood: `logLik`
   - AIC, BIC
   - KS statistic between empirical CDF and model CDF
   - MSE between empirical CDF and model CDF (as a simple goodness-of-fit metric)

4. Export summary tables similar to the ones in the Note:
   - `results/single_exp_fits.csv`
   - `results/biexp_fits.csv`

5. (Nice-to-have) Produce diagnostic plots per time point:
   - histogram + fitted PDF (single vs bi-exponential)
   - empirical CDF vs model CDF (for KS visualisation)

## Implementation tasks (R)

Please implement this in **R** with tidyverse:

1. Create a script: `R/fit_truncated_exponential_models.R`.
2. In this script:
   - Read `data/H22_VOLUME_LS_v2.csv`.
   - Clean column names and keep lesions with `Volume_um3 >= v_min`.
   - For each time point, fit:
     - truncated single exponential model using a closed-form MLE for θ
     - truncated bi-exponential mixture model using multi-start BFGS
   - Compute logLik, AIC, BIC, KS, MSE.
   - Save summary tables to the `results/` folder.
3. Make sure functions are modular:
   - `fit_trunc_single_exp(v, vmin)`
   - `fit_trunc_bi_exp(v, vmin, n_start = 20)`
   - `gof_trunc_model(v, cdf_fun, vmin)` for KS and MSE
4. Add short comments explaining how the truncation at `v_min` is handled in both models.
