# Predictor level encoding
# X should contain non-negative integer codes 0, ..., d_j - 1 per column.
build_predictor_levels <- function(X) {
  d_j <- apply(X, 2, max) + 1L
  max_d <- max(d_j)
  X_ell_list <- lapply(0:(max_d - 1), \(ell) 1 * (X == ell))
  n_j_ell_list <- lapply(X_ell_list, colSums)
  active_list <- lapply(0:(max_d - 1), \(ell) d_j > ell)
  list(
    d_j = d_j,
    X_ell_list = X_ell_list,
    n_j_ell_list = n_j_ell_list,
    active_list = active_list
  )
}

# Shared BF_{j1} for all p predictors (kernel-independent)
compute_log_bf1 <- function(
  C,
  w,
  X_ell_list,
  n_j_ell_list,
  active_list,
  tau_omega
) {
  # C should be a sparse matrix (Matrix package) for performance
  k <- length(w)
  p <- length(n_j_ell_list[[1]])
  n_h <- diff(C@p) # fast column counts for dgCMatrix

  first_term <- -sum(n_h * log(w))

  prior_alpha <- tau_omega * w
  lgamma_prior <- sum(lgamma(prior_alpha))
  second_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    n_jh_ell <- as.matrix(crossprod(C, X_ell_list[[ell_idx]]))
    contrib <- colSums(lgamma(n_jh_ell + prior_alpha)) -
      lgamma_prior -
      lgamma(n_j_ell_list[[ell_idx]] + tau_omega) +
      lgamma(tau_omega)
    contrib[!active_list[[ell_idx]]] <- 0
    second_term <- second_term + contrib
  }

  first_term + second_term
}

# BF_{j2} for Gaussian kernel
log_det <- function(x) {
  as.numeric(determinant(x)$modulus)
}

compute_log_bf2_gaussian <- function(
  C,
  mu,
  Sigma,
  Y,
  X_ell_list,
  active_list,
  tau_mu,
  tau_sigma,
  d_j
) {
  k <- length(mu)
  p <- length(d_j)
  q <- ncol(Y)
  n_h <- diff(C@p)

  # Cluster quantities
  log_det_Sigma <- sapply(Sigma, log_det)
  log_det_tau_Sigma <- q * log(tau_sigma) + log_det_Sigma

  # Shared term (doesn't depend on j):
  # sum_h [n_h/2 * log|Sigma_h| + 1/2 * sum quad forms]
  shared_term <- 0
  for (h in seq_len(k)) {
    if (n_h[h] == 0) {
      next
    }
    idx_h <- which(C[, h] == 1)
    centered <- sweep(Y[idx_h, , drop = FALSE], 2, mu[[h]])
    quad <- rowSums((centered %*% solve(Sigma[[h]])) * centered)
    shared_term <- shared_term +
      n_h[h] / 2 * log_det_Sigma[h] +
      sum(quad) / 2
  }

  # Constant term (doesn't depend on j):
  # sum_{ell,h} n_{jh}(ell) * q/2 * log(2) = n * q/2 * log(2) for all j
  constant_term <- sum(n_h) * q / 2 * log(2)

  # Prior per level (doesn't depend on j or ell):
  # sum_h [tau_sigma/2 * log|tau_sigma * Sigma_h| + q/2 * log(tau_mu)
  #        - lgamma_q(tau_sigma / 2)]
  lgamma_q_prior <- lmvgamma(tau_sigma / 2, q)
  prior_per_level <- tau_sigma /
    2 *
    sum(log_det_tau_Sigma) +
    k * q / 2 * log(tau_mu) -
    k * lgamma_q_prior

  # Data term (vectorized over j):
  # sum_ell sum_h [lgamma_q((tau_sigma + n_{jh})/2)
  #               - (tau_sigma + n_{jh})/2 * log|Lambda_{jh}|
  #               - q/2 * log(tau_mu + n_{jh})]
  spec_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    X_ell <- X_ell_list[[ell_idx]]
    n_jh_ell <- as.matrix(crossprod(C, X_ell)) # k x p

    for (h in seq_len(k)) {
      S_jh <- crossprod(C[, h] * Y, X_ell) # q x p
      SS_jh <- array(0, c(q, q, p))
      for (a in seq_len(q)) {
        for (b in a:q) {
          v <- drop(crossprod(C[, h] * Y[, a] * Y[, b], X_ell))
          SS_jh[a, b, ] <- v
          if (b > a) SS_jh[b, a, ] <- v
        }
      }

      safe_n <- pmax(n_jh_ell[h, ], 1)
      y_bar <- S_jh / rep(safe_n, each = q)
      dev <- y_bar - mu[[h]]
      kf <- tau_mu * n_jh_ell[h, ] / (tau_mu + safe_n)
      tau_Sigma_h <- tau_sigma * Sigma[[h]]
      Lambda <- array(0, c(q, q, p))
      for (a in seq_len(q)) {
        for (b in a:q) {
          W_ab <- SS_jh[a, b, ] - n_jh_ell[h, ] * y_bar[a, ] * y_bar[b, ]
          D_ab <- kf * dev[a, ] * dev[b, ]
          Lambda[a, b, ] <- tau_Sigma_h[a, b] + W_ab + D_ab
          if (b > a) Lambda[b, a, ] <- Lambda[a, b, ]
        }
      }

      log_det_Lambda <- numeric(p)
      if (q == 1L) {
        log_det_Lambda <- log(Lambda[1, 1, ])
      } else {
        for (j in which(active_list[[ell_idx]] & n_jh_ell[h, ] > 0)) {
          log_det_Lambda[j] <- log_det(Lambda[,, j])
        }
      }

      lgamma_q_post <- lmvgamma((tau_sigma + n_jh_ell[h, ]) / 2, q)

      contrib <- lgamma_q_post -
        (tau_sigma + n_jh_ell[h, ]) / 2 * log_det_Lambda -
        q / 2 * log(tau_mu + n_jh_ell[h, ])
      contrib[n_jh_ell[h, ] == 0] <- 0
      contrib[!active_list[[ell_idx]]] <- 0

      spec_term <- spec_term + contrib
    }
  }

  shared_term + constant_term + d_j * prior_per_level + spec_term
}

# log BF_{j2} for categorical kernel
compute_log_bf2_multinomial <- function(
  C,
  psi,
  Y,
  X_ell_list,
  active_list,
  CY_sparse,
  tau_psi,
  d_j
) {
  # psi: list of length q, psi[[t]] is a k x m_t matrix
  # Y: n x q with y_{it} in {1, ..., m_t}
  # C should be a sparse matrix (Matrix package)
  # CY_sparse[[h]][[t]][[s]] is a sparse n×1 matrix = C[,h] * (Y[,t]==s)
  k <- ncol(C)
  p <- length(active_list[[1]])
  q <- length(CY_sparse[[1]])
  m <- sapply(psi, ncol)

  # Shared term (doesn't depend on j):
  # sum_h sum_t sum_s n_{hts} * log(psi_{hts})
  shared_term <- 0
  for (t in seq_len(q)) {
    for (h in seq_len(k)) {
      n_hts <- tabulate(Y[C[, h] == 1, t], nbins = m[t])
      shared_term <- shared_term + sum(n_hts * log(psi[[t]][h, ]))
    }
  }

  # Prior per level (doesn't depend on j or ell):
  # sum_h sum_t [sum_s lgamma(tau_psi * psi_{hts}) - lgamma(tau_psi)]
  prior_per_level <- 0
  for (t in seq_len(q)) {
    for (h in seq_len(k)) {
      prior_per_level <- prior_per_level +
        sum(lgamma(tau_psi * psi[[t]][h, ])) -
        lgamma(tau_psi)
    }
  }

  # Data term (vectorized over j)
  spec_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    X_ell <- X_ell_list[[ell_idx]]
    n_jh_ell <- as.matrix(crossprod(C, X_ell))

    contrib <- numeric(p)
    for (h in seq_len(k)) {
      for (t in seq_len(q)) {
        for (si in seq_len(m[t])) {
          n_jhts_ell <- drop(crossprod(CY_sparse[[h]][[t]][[si]], X_ell))
          contrib <- contrib +
            lgamma(n_jhts_ell + tau_psi * psi[[t]][h, si])
        }
        contrib <- contrib - lgamma(n_jh_ell[h, ] + tau_psi)
      }
    }
    contrib[!active_list[[ell_idx]]] <- 0
    spec_term <- spec_term + contrib
  }

  spec_term - shared_term - d_j * prior_per_level
}

# log BF_{j2} for Poisson kernel
compute_log_bf2_poisson <- function(
  C,
  rho,
  Y,
  X_ell_list,
  active_list,
  CY,
  tau_rho,
  d_j
) {
  # rho is a k x q matrix (rho[h, t] = rho_{ht})
  # Y is n x q with count responses
  # X_ell_list[[ell+1]] = 1*(X == ell), active_list[[ell+1]] = d_j > ell
  # CY[[h]] = C[,h] * Y (n x q matrix), precomputed outside
  k <- ncol(C)
  p <- length(d_j)
  q <- ncol(Y)
  n_h <- diff(C@p)

  # Shared term (doesn't depend on j):
  # sum_h sum_t [n_h * rho_{ht} - S_{ht} * log(rho_{ht})]
  shared_term <- 0
  for (h in seq_len(k)) {
    S_ht <- colSums(CY[[h]]) # q-vector
    shared_term <- shared_term +
      sum(n_h[h] * rho[h, ] - S_ht * log(rho[h, ]))
  }

  # Prior per level (doesn't depend on j or ell):
  # sum_h sum_t [tau_rho * rho_{ht} * log(tau_rho) - lgamma(tau_rho * rho_{ht})]
  prior_per_level <- sum(tau_rho * rho * log(tau_rho) - lgamma(tau_rho * rho))

  # Data term (vectorized over j):
  spec_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    X_ell <- X_ell_list[[ell_idx]]
    n_jh_ell <- as.matrix(crossprod(C, X_ell)) # k x p

    contrib <- numeric(p)
    for (h in seq_len(k)) {
      log_tau_plus_n <- log(tau_rho + n_jh_ell[h, ])
      for (t in seq_len(q)) {
        S_jht_ell <- drop(crossprod(CY[[h]][, t], X_ell)) # p-vector
        a <- tau_rho * rho[h, t]
        contrib <- contrib +
          lgamma(a + S_jht_ell) -
          (a + S_jht_ell) * log_tau_plus_n
      }
    }
    contrib[!active_list[[ell_idx]]] <- 0
    spec_term <- spec_term + contrib
  }

  shared_term + d_j * prior_per_level + spec_term
}

# General function to compute log Bayes factors
compute_log_bfs <- function(
  fit,
  Y_N,
  Y_C,
  Y_P,
  X,
  tau_omega,
  tau_mu = NULL,
  tau_Sigma = NULL,
  tau_psi = NULL,
  tau_rho = NULL
) {
  S <- nrow(fit$z)
  n <- ncol(fit$z)
  k <- ncol(fit$w)

  qN <- if (is.null(Y_N)) 0L else ncol(Y_N)
  qC <- if (is.null(Y_C)) 0L else ncol(Y_C)
  qP <- if (is.null(Y_P)) 0L else ncol(Y_P)

  # Predictor level encoding (shared across samples and kernels)
  levels <- build_predictor_levels(X)
  d_j <- levels$d_j
  X_ell_list <- levels$X_ell_list
  n_j_ell_list <- levels$n_j_ell_list
  active_list <- levels$active_list
  p <- length(d_j)

  # Categorical indicators (shared across samples):
  # qC-list of m_t-lists of length-n indicator vectors
  if (qC > 0) {
    m <- sapply(fit$psi, \(x) dim(x)[3])
    Y_ind <- lapply(seq_len(qC), \(t) {
      lapply(seq_len(m[t]), \(s) as.numeric(Y_C[, t] == s))
    })
  }

  results <- lapply(
    seq_len(S),
    function(s) {
      z_s <- fit$z[s, ]
      C <- Matrix::sparseMatrix(i = seq_len(n), j = z_s, x = 1, dims = c(n, k))
      w_s <- fit$w[s, ]

      log_bf1_s <- compute_log_bf1(
        C,
        w_s,
        X_ell_list,
        n_j_ell_list,
        active_list,
        tau_omega
      )

      log_bf2_s <- numeric(p)

      if (qN > 0) {
        mu_s <- lapply(seq_len(k), \(h) fit$mu[s, h, ])
        Sigma_s <- lapply(seq_len(k), \(h) {
          M <- fit$Sigma[s, h, , , drop = FALSE]
          dim(M) <- c(qN, qN)
          M
        })
        log_bf2_s <- log_bf2_s +
          compute_log_bf2_gaussian(
            C,
            mu_s,
            Sigma_s,
            Y_N,
            X_ell_list,
            active_list,
            tau_mu,
            tau_Sigma,
            d_j
          )
      }

      if (qC > 0) {
        psi_s <- lapply(fit$psi, \(x) matrix(x[s, , ], nrow = k))
        CY_sparse <- lapply(seq_len(k), \(h) {
          lapply(seq_len(qC), \(t) {
            lapply(seq_len(m[t]), \(si) {
              Matrix::Matrix(C[, h] * Y_ind[[t]][[si]], ncol = 1, sparse = TRUE)
            })
          })
        })
        log_bf2_s <- log_bf2_s +
          compute_log_bf2_multinomial(
            C,
            psi_s,
            Y_C,
            X_ell_list,
            active_list,
            CY_sparse,
            tau_psi,
            d_j
          )
      }

      if (qP > 0) {
        rho_s <- matrix(fit$rho[s, , ], nrow = k)
        CY <- lapply(seq_len(k), \(h) C[, h] * Y_P) # list of n x qP matrices
        log_bf2_s <- log_bf2_s +
          compute_log_bf2_poisson(
            C,
            rho_s,
            Y_P,
            X_ell_list,
            active_list,
            CY,
            tau_rho,
            d_j
          )
      }

      list(log_bf1 = log_bf1_s, log_bf2 = log_bf2_s)
    }
  )

  log_bf1 <- do.call(rbind, lapply(results, `[[`, "log_bf1"))
  log_bf2 <- do.call(rbind, lapply(results, `[[`, "log_bf2"))
  list(log_bf1 = log_bf1, log_bf2 = log_bf2)
}
