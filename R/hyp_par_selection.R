# Implementation of Supplemental D. For the mixed kernel, selection is done
# separately for each block of theta (Gaussian, multinomial, Poisson) using
# the common mixture fit.

# Multivariate log-Gamma function (also used in bayes_factors.R)
lmvgamma <- function(a, q) {
  out <- q * (q - 1) / 4 * log(pi)
  for (t in seq_len(q)) {
    out <- out + lgamma(a + (1 - t) / 2)
  }
  out
}

# Mixture weights ----------------------------------------------------------
select_tau_omega <- function(w_samples, interval_w = c(1e-3, 1e4)) {
  S <- nrow(w_samples)
  k <- ncol(w_samples)
  log_w_sum <- sum(log(w_samples))

  obj_function <- function(tau_w) {
    t1 <- log_w_sum * tau_w / k
    t2 <- S * lgamma(tau_w)
    t3 <- S * k * lgamma(tau_w / k)
    t1 + t2 - t3
  }

  stats::optimize(obj_function, interval_w, maximum = TRUE)$maximum
}

# Gaussian selection --------------------------------------------------------
select_tau_gaussian <- function(
  mu_samples,
  Sigma_samples,
  interval_Sigma = c(1e-3, 1e4)
) {
  S <- dim(mu_samples)[1]
  k <- dim(mu_samples)[2]
  Sk <- S * k
  q <- dim(mu_samples)[3]

  # Per-sample inverses and log-determinants
  Sigma_inv_samples <- apply(Sigma_samples, c(1, 2), solve)
  dim(Sigma_inv_samples) <- c(q, q, S, k) # for the case q = 1
  Sigma_logdet_samples <- apply(Sigma_samples, c(1, 2), \(S) {
    as.numeric(determinant(S)$modulus)
  })

  # Quadratic sum
  mu_bar <- apply(mu_samples, 3, mean) # q dim
  centered <- sweep(mu_samples, 3, mu_bar) # S, k, q
  quad_sum <- 0
  for (s in seq_len(S)) {
    for (h in seq_len(k)) {
      d <- centered[s, h, ]
      quad_sum <- quad_sum + t(d) %*% Sigma_inv_samples[,, s, h] %*% d
    }
  }

  # tau_mu in closed form
  tau_mu <- Sk * q / quad_sum

  # Optimization over tau_Sigma
  Sigma_bar <- apply(Sigma_samples, c(3, 4), mean)
  trace_sum <- apply(Sigma_inv_samples, c(3, 4), \(S) {
    sum(diag(Sigma_bar %*% S))
  }) |>
    sum()
  log_det_sum <- sum(Sigma_logdet_samples)
  log_det_half_Sigma_bar <- as.numeric(determinant(Sigma_bar / 2)$modulus)
  c0 <- Sk * log_det_half_Sigma_bar - log_det_sum - trace_sum

  obj_function <- function(tau_Sigma) {
    (Sk * q * tau_Sigma * log(tau_Sigma)) +
      c0 * tau_Sigma -
      2 * Sk * lmvgamma(tau_Sigma / 2, q)
  }

  tau_Sigma <- stats::optimize(
    obj_function,
    interval_Sigma,
    maximum = TRUE
  )$maximum
  c(drop(tau_mu), tau_Sigma)
}

# Multinomial selection -----------------------------------------------------
# psi_samples: list of length q, each element an (S, k, m_t) array
select_tau_multinomial <- function(psi_samples, interval_psi = c(1e-3, 1e4)) {
  S <- dim(psi_samples[[1]])[1]
  k <- dim(psi_samples[[1]])[2]
  q <- length(psi_samples)
  Sk <- S * k

  c1 <- 0
  psi_bar_all <- numeric(0) # Used for the second term
  for (t in seq_len(q)) {
    psi_t <- psi_samples[[t]] # (S, k, m_t)
    psi_bar_t <- apply(psi_t, 3, mean) # length m_t
    c1 <- c1 + sum(psi_bar_t * apply(log(psi_t), 3, sum))
    psi_bar_all <- c(psi_bar_all, psi_bar_t)
  }

  obj_function <- function(tau_psi) {
    t1 <- c1 * tau_psi
    t2 <- Sk * (q * lgamma(tau_psi) - sum(lgamma(psi_bar_all * tau_psi)))
    t1 + t2
  }

  stats::optimize(obj_function, interval_psi, maximum = TRUE)$maximum
}

# Poisson selection ---------------------------------------------------------
# rho_samples: array (S, k, q) of Gamma rate parameters
select_tau_poisson <- function(rho_samples, interval_rho = c(1e-3, 1e4)) {
  S <- dim(rho_samples)[1]
  k <- dim(rho_samples)[2]
  q <- dim(rho_samples)[3]
  Sk <- S * k

  rho_bar <- apply(rho_samples, 3, mean)
  c1 <- Sk * sum(rho_bar)

  log_rho <- log(rho_samples)
  c2 <- 0
  for (t in seq_len(q)) {
    c2 <- c2 + rho_bar[t] * sum(log_rho[,, t]) - sum(rho_samples[,, t])
  }

  obj_function <- function(tau_rho) {
    t1 <- c1 * tau_rho * log(tau_rho)
    t2 <- Sk * sum(lgamma(rho_bar * tau_rho))
    t3 <- c2 * tau_rho

    t1 - t2 + t3
  }

  stats::optimize(obj_function, interval_rho, maximum = TRUE)$maximum
}
