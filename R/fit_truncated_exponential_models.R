# R/fit_truncated_exponential_models.R

library(tidyverse)

# ---------- Utilities ----------

softplus <- function(x) log1p(exp(x))
inv_logit <- function(x) 1 / (1 + exp(-x))

# Goodness-of-fit: KS and MSE on CDF
gof_trunc_model <- function(v, cdf_fun, vmin) {
  v <- sort(v[v >= vmin])
  n <- length(v)
  if (n == 0) stop("No data >= vmin for GOF")

  emp_cdf <- (1:n) / n
  model_cdf <- cdf_fun(v)

  ks  <- max(abs(emp_cdf - model_cdf))
  mse <- mean((emp_cdf - model_cdf)^2)

  list(KS = ks, MSE = mse)
}

# ---------- 1) Truncated single exponential model ----------

# Model: f(v) = 1/theta * exp(-(v - vmin)/theta), v >= vmin
# Equivalent to standard Exp(theta) on x = v - vmin.

fit_trunc_single_exp <- function(v, vmin) {
  v <- v[v >= vmin]
  n <- length(v)
  if (n == 0) stop("No data >= vmin for single-exp fit")

  x <- v - vmin
  theta_hat <- mean(x)

  # log-likelihood under Exp(mean = theta_hat) on x
  loglik <- sum(dexp(x, rate = 1 / theta_hat, log = TRUE))

  list(theta = theta_hat, logLik = loglik, n = n)
}

# ---------- 2) Truncated bi-exponential mixture model ----------

# Untruncated mixture:
# f(v) = a1 * (1/v0) * exp(-v / v0) + a2 * (1/v1) * exp(-v / v1)
# with a1 + a2 = 1, v0 > 0, v1 > v0
#
# Truncation at vmin:
# f_trunc(v) = f(v) / P(V >= vmin)
# P(V >= vmin) = a1 * exp(-vmin/v0) + a2 * exp(-vmin/v1)
#
# Reparameterisation:
#   a2 = inv_logit(w), a1 = 1 - a2
#   v0 = softplus(u)
#   v1 = v0 + softplus(z)

negloglik_trunc_bi_exp <- function(par, v, vmin) {
  w <- par[1]
  u <- par[2]
  z <- par[3]

  a2 <- inv_logit(w)
  a1 <- 1 - a2

  v0 <- softplus(u)
  v1 <- v0 + softplus(z)

  if (v0 <= 0 || v1 <= 0 || a1 <= 0 || a2 <= 0) {
    return(1e10)
  }

  # only use v >= vmin (should already be filtered)
  v <- v[v >= vmin]

  # numerator: mixture density at v
  num <- a1 * (1 / v0) * exp(-v / v0) +
         a2 * (1 / v1) * exp(-v / v1)

  # survival at vmin: P(V >= vmin)
  surv_vmin <- a1 * exp(-vmin / v0) + a2 * exp(-vmin / v1)

  if (surv_vmin <= 0 || any(num <= 0) || any(!is.finite(num))) {
    return(1e10)
  }

  dens <- num / surv_vmin

  if (any(dens <= 0) || any(!is.finite(dens))) {
    return(1e10)
  }

  -sum(log(dens))
}

fit_trunc_bi_exp <- function(v, vmin, n_start = 20) {
  v <- v[v >= vmin]
  n <- length(v)
  if (n == 0) stop("No data >= vmin for bi-exp fit")

  best_opt <- NULL
  best_val <- Inf

  for (i in seq_len(n_start)) {
    par0 <- rnorm(3, mean = 0, sd = 1)
    opt <- try(
      optim(
        par0,
        negloglik_trunc_bi_exp,
        v = v,
        vmin = vmin,
        method = "BFGS"
      ),
      silent = TRUE
    )

    if (inherits(opt, "try-error")) next

    if (opt$convergence == 0 && opt$value < best_val) {
      best_val <- opt$value
      best_opt <- opt
    }
  }

  if (is.null(best_opt)) stop("Bi-exponential optimisation failed in all starts")

  par <- best_opt$par
  w <- par[1]
  u <- par[2]
  z <- par[3]

  a2 <- inv_logit(w)
  a1 <- 1 - a2
  v0 <- softplus(u)
  v1 <- v0 + softplus(z)

  logLik <- -best_opt$value

  list(a1 = a1, a2 = a2, v0 = v0, v1 = v1,
       logLik = logLik, par = par, n = n)
}

# Truncated bi-exponential CDF for GOF:
cdf_trunc_bi <- function(v, a1, a2, v0, v1, vmin) {
  # Untruncated mixture CDF
  F_u <- 1 - a1 * exp(-v / v0) - a2 * exp(-v / v1)
  F_u_min <- 1 - a1 * exp(-vmin / v0) - a2 * exp(-vmin / v1)
  # Truncated at vmin
  (F_u - F_u_min) / (1 - F_u_min)
}

# ---------- 3) Main pipeline ----------

vmin <- 10

# Adjust the column names according to your real CSV:
df <- read_csv("data/H22_VOLUME_LS_v2.csv") %>%
  rename(
    TimePoint = `TimePoint (Day)`,
    Volume_um3 = Volume_um3   # or `Volume_um³` if that's the actual name
  ) %>%
  mutate(
    TimePoint = as.character(TimePoint)
  )

timepoints <- sort(unique(df$TimePoint))

single_results <- list()
bi_results <- list()

for (tp in timepoints) {
  cat("Fitting time point:", tp, "\n")

  v <- df %>%
    filter(TimePoint == tp) %>%
    pull(Volume_um3)

  v <- v[!is.na(v)]

  # ----- Single-exponential -----
  fit1 <- fit_trunc_single_exp(v, vmin)
  theta <- fit1$theta
  loglik1 <- fit1$logLik
  n      <- fit1$n
  k1     <- 1  # number of parameters

  AIC1 <- 2 * k1 - 2 * loglik1
  BIC1 <- k1 * log(n) - 2 * loglik1

  cdf_single <- function(x) {
    # CDF: F(v) = 1 - exp(-(v - vmin)/theta), v >= vmin
    1 - exp(-(x - vmin) / theta)
  }

  gof1 <- gof_trunc_model(v, cdf_single, vmin)

  single_results[[tp]] <- tibble(
    TimePoint = tp,
    theta     = theta,
    logLik    = loglik1,
    AIC       = AIC1,
    BIC       = BIC1,
    KS        = gof1$KS,
    MSE       = gof1$MSE
  )

  # ----- Bi-exponential mixture -----
  fit2 <- fit_trunc_bi_exp(v, vmin, n_start = 20)
  a1 <- fit2$a1
  a2 <- fit2$a2
  v0 <- fit2$v0
  v1 <- fit2$v1
  loglik2 <- fit2$logLik
  n2 <- fit2$n
  k2 <- 3  # (a2, v0, v1) effectively three free parameters

  AIC2 <- 2 * k2 - 2 * loglik2
  BIC2 <- k2 * log(n2) - 2 * loglik2

  cdf_bi <- function(x) cdf_trunc_bi(x, a1, a2, v0, v1, vmin)
  gof2 <- gof_trunc_model(v, cdf_bi, vmin)

  bi_results[[tp]] <- tibble(
    TimePoint = tp,
    a2        = a2,
    v0        = v0,
    v1        = v1,
    logLik    = loglik2,
    AIC       = AIC2,
    BIC       = BIC2,
    KS        = gof2$KS,
    MSE       = gof2$MSE
  )
}

single_tbl <- bind_rows(single_results)
bi_tbl     <- bind_rows(bi_results)

if (!dir.exists("results")) dir.create("results")

write_csv(single_tbl, "results/single_exp_fits.csv")
write_csv(bi_tbl,     "results/biexp_fits.csv")

print(single_tbl)
print(bi_tbl)
