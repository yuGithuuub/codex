import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize


# ---------- Utils ----------

def softplus(x):
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def gof_trunc_model(v, cdf_fun, vmin):
    """
    KS + MSE on CDF, for v >= vmin.
    v : 1D array of volumes
    cdf_fun : function(x) -> model CDF at x
    """
    v = np.asarray(v)
    v = v[v >= vmin]
    v = np.sort(v)
    n = len(v)
    if n == 0:
        raise ValueError("No data >= vmin for GOF")

    emp_cdf = np.arange(1, n + 1, dtype=float) / n
    model_cdf = cdf_fun(v)

    diffs = emp_cdf - model_cdf
    ks = np.max(np.abs(diffs))
    mse = np.mean(diffs ** 2)

    return ks, mse


# ---------- 1) Truncated single exponential model ----------

def fit_trunc_single_exp(v, vmin):
    """
    Truncated single exponential at vmin.
    Model on x = v - vmin ~ Exp(theta).
    """
    v = np.asarray(v)
    v = v[v >= vmin]
    if len(v) == 0:
        raise ValueError("No data >= vmin for single-exp fit")

    x = v - vmin
    theta = float(np.mean(x))

    # log-likelihood under Exp(mean=theta) for x
    # f(x) = 1/theta * exp(-x/theta)
    n = len(x)
    loglik = -n * np.log(theta) - np.sum(x) / theta

    return {
        "theta": theta,
        "logLik": loglik,
        "n": n
    }


# ---------- 2) Truncated bi-exponential mixture model ----------

def negloglik_trunc_bi_exp(params, v, vmin):
    """
    Negative log-likelihood for truncated bi-exponential mixture.
    params = [w, u, z]:
        a2 = inv_logit(w), a1 = 1 - a2
        v0 = softplus(u)
        v1 = v0 + softplus(z)
    """
    w, u, z = params

    a2 = inv_logit(w)
    a1 = 1.0 - a2

    v0 = softplus(u)
    v1 = v0 + softplus(z)

    # basic constraints
    if v0 <= 0 or v1 <= 0 or a1 <= 0 or a2 <= 0:
        return 1e10

    v = np.asarray(v)
    v = v[v >= vmin]
    if len(v) == 0:
        return 1e10

    # untruncated mixture density at v
    num = a1 * (1.0 / v0) * np.exp(-v / v0) + \
          a2 * (1.0 / v1) * np.exp(-v / v1)

    # survival at vmin: P(V >= vmin)
    surv_vmin = a1 * np.exp(-vmin / v0) + a2 * np.exp(-vmin / v1)

    if surv_vmin <= 0 or np.any(num <= 0) or not np.isfinite(surv_vmin):
        return 1e10

    dens = num / surv_vmin
    if np.any(dens <= 0) or not np.all(np.isfinite(dens)):
        return 1e10

    return -np.sum(np.log(dens))


def fit_trunc_bi_exp(v, vmin, n_start=20):
    """
    Multi-start BFGS for truncated bi-exponential mixture.
    """
    v = np.asarray(v)
    v = v[v >= vmin]
    if len(v) == 0:
        raise ValueError("No data >= vmin for bi-exp fit")

    best_res = None
    best_val = np.inf

    for _ in range(n_start):
        par0 = np.random.normal(size=3)
        try:
            res = minimize(
                negloglik_trunc_bi_exp,
                par0,
                args=(v, vmin),
                method="BFGS"
            )
        except Exception:
            continue

        if not res.success:
            continue

        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    if best_res is None:
        raise RuntimeError("Bi-exponential optimisation failed in all starts")

    w, u, z = best_res.x
    a2 = inv_logit(w)
    a1 = 1.0 - a2
    v0 = softplus(u)
    v1 = v0 + softplus(z)

    loglik = -best_res.fun

    return {
        "a1": a1,
        "a2": a2,
        "v0": v0,
        "v1": v1,
        "logLik": loglik,
        "n": len(v),
        "params": best_res.x
    }


def cdf_trunc_single(v, theta, vmin):
    """
    Truncated single-exponential CDF:
    F(v) = 1 - exp(-(v - vmin)/theta), v >= vmin
    (equivalent to Exp on x = v - vmin)
    """
    v = np.asarray(v)
    x = v - vmin
    return 1.0 - np.exp(-x / theta)


def cdf_trunc_bi(v, a1, a2, v0, v1, vmin):
    """
    Truncated bi-exponential mixture CDF.
    Untruncated mixture CDF:
      F_u(v) = 1 - a1 * exp(-v / v0) - a2 * exp(-v / v1)
    Truncated at vmin:
      F_trunc(v) = (F_u(v) - F_u(vmin)) / (1 - F_u(vmin))
    """
    v = np.asarray(v)

    F_u = 1.0 - a1 * np.exp(-v / v0) - a2 * np.exp(-v / v1)
    F_u_min = 1.0 - a1 * np.exp(-vmin / v0) - a2 * np.exp(-vmin / v1)

    denom = 1.0 - F_u_min
    return (F_u - F_u_min) / denom


# ---------- 3) Main pipeline ----------

def main():
    vmin = 10.0  # µm³ detection threshold

    df = pd.read_csv("data/H22_VOLUME_LS_v2.csv")

    # normalise column names
    if "TimePoint (Day)" in df.columns:
        df = df.rename(columns={"TimePoint (Day)": "TimePoint"})

    if "Volume_um3" in df.columns:
        df = df.rename(columns={"Volume_um3": "Volume_um3"})
    elif "Volume_um³" in df.columns:
        # if your column actually uses the µ character
        df = df.rename(columns={"Volume_um³": "Volume_um3"})
    else:
        raise ValueError("Cannot find volume column: expected 'Volume_um3' or 'Volume_um³'")

    df = df.dropna(subset=["Volume_um3", "TimePoint"])

    timepoints = sorted(df["TimePoint"].unique())

    single_rows = []
    bi_rows = []

    for tp in timepoints:
        print(f"Fitting time point: {tp}")
        v = df.loc[df["TimePoint"] == tp, "Volume_um3"].values
        v = v[np.isfinite(v)]
        v = v[v >= vmin]
        if len(v) == 0:
            print(f"  No data >= vmin for {tp}, skipping.")
            continue

        # --- single exponential ---
        sfit = fit_trunc_single_exp(v, vmin)
        theta = sfit["theta"]
        loglik1 = sfit["logLik"]
        n = sfit["n"]
        k1 = 1  # theta

        AIC1 = 2 * k1 - 2 * loglik1
        BIC1 = k1 * np.log(n) - 2 * loglik1

        ks1, mse1 = gof_trunc_model(
            v,
            lambda x: cdf_trunc_single(x, theta, vmin),
            vmin
        )

        single_rows.append({
            "TimePoint": tp,
            "theta": theta,
            "logLik": loglik1,
            "AIC": AIC1,
            "BIC": BIC1,
            "KS": ks1,
            "MSE": mse1,
            "n": n
        })

        # --- bi-exponential mixture ---
        bfit = fit_trunc_bi_exp(v, vmin, n_start=20)
        a1 = bfit["a1"]
        a2 = bfit["a2"]
        v0 = bfit["v0"]
        v1 = bfit["v1"]
        loglik2 = bfit["logLik"]
        n2 = bfit["n"]
        k2 = 3  # a2, v0, v1 (reparameterised by w,u,z)

        AIC2 = 2 * k2 - 2 * loglik2
        BIC2 = k2 * np.log(n2) - 2 * loglik2

        ks2, mse2 = gof_trunc_model(
            v,
            lambda x: cdf_trunc_bi(x, a1, a2, v0, v1, vmin),
            vmin
        )

        bi_rows.append({
            "TimePoint": tp,
            "a2": a2,   # slow/escape fraction
            "v0": v0,   # fast/clearance scale
            "v1": v1,   # escape/large scale
            "logLik": loglik2,
            "AIC": AIC2,
            "BIC": BIC2,
            "KS": ks2,
            "MSE": mse2,
            "n": n2
        })

    single_df = pd.DataFrame(single_rows)
    bi_df = pd.DataFrame(bi_rows)

    Path("results").mkdir(exist_ok=True)
    single_df.to_csv("results/single_exp_fits_py.csv", index=False)
    bi_df.to_csv("results/biexp_fits_py.csv", index=False)

    print("\nSingle exponential fits:")
    print(single_df)
    print("\nBi-exponential mixture fits:")
    print(bi_df)


if __name__ == "__main__":
    main()
