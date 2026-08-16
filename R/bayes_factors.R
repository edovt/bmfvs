# Predictor level encoding
# X should contain non-negative integer codes 0, ..., d_j - 1 per column.
#
# `X_ell_list` holds max_d dense n x p indicator matrices, which is the largest
# object in the whole procedure; it is only built for the "r" backend, since
# the C++ backend reads the integer X directly.
build_predictor_levels <- function(X, indicators = TRUE) {
  d_j <- apply(X, 2, max) + 1L
  max_d <- max(d_j)
  active_list <- lapply(0:(max_d - 1), \(ell) d_j > ell)
  out <- list(d_j = d_j, max_d = max_d, active_list = active_list)
  if (indicators) {
    out$X_ell_list <- lapply(0:(max_d - 1), \(ell) 1 * (X == ell))
    out$n_j_ell_list <- lapply(out$X_ell_list, colSums)
  }
  out
}

# n_{jh}(ell) for every level: a list of k x p matrices. Shared by all four
# Bayes factors, so it is computed once per posterior sample.
compute_level_counts <- function(C, X_ell_list) {
  lapply(X_ell_list, \(X_ell) as.matrix(Matrix::crossprod(C, X_ell)))
}

# Shared BF_{j1} for all p predictors (kernel-independent)
compute_log_bf1 <- function(
  C,
  w,
  n_j_ell_list,
  active_list,
  tau_omega,
  n_jh_ell_list
) {
  # C should be a sparse matrix (Matrix package) for performance
  p <- length(n_j_ell_list[[1]])
  n_h <- diff(C@p) # fast column counts for dgCMatrix

  first_term <- -sum(n_h * log(w))

  prior_alpha <- tau_omega * w
  lgamma_prior <- sum(lgamma(prior_alpha))
  second_term <- numeric(p)
  for (ell_idx in seq_along(n_jh_ell_list)) {
    n_jh_ell <- n_jh_ell_list[[ell_idx]]
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
# All the matrices whose determinant is needed here (Sigma_h, Lambda_{jh}) are
# symmetric positive definite, so a Cholesky is both cheaper and more accurate
# than the LU factorization behind determinant().
log_det <- function(x) {
  r <- tryCatch(chol(x), error = function(e) NULL)
  if (is.null(r)) {
    return(as.numeric(determinant(x)$modulus))
  }
  2 * sum(log(diag(r)))
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
  d_j,
  n_jh_ell_list,
  idx_by_h
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
    idx_h <- idx_by_h[[h]]
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
  # P_h = tau_sigma/2 * log|tau_sigma * Sigma_h| + q/2 * log(tau_mu)
  #       - lgamma_q(tau_sigma / 2)
  lgamma_q_prior <- lmvgamma(tau_sigma / 2, q)
  P_h <- tau_sigma / 2 * log_det_tau_Sigma + q / 2 * log(tau_mu) -
    lgamma_q_prior
  prior_per_level <- sum(P_h)

  # Data term (vectorized over j):
  # sum_ell sum_h [lgamma_q((tau_sigma + n_{jh})/2)
  #               - (tau_sigma + n_{jh})/2 * log|Lambda_{jh}|
  #               - q/2 * log(tau_mu + n_{jh})]
  spec_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    X_ell <- X_ell_list[[ell_idx]]
    n_jh_ell <- n_jh_ell_list[[ell_idx]] # k x p

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
      # An empty cell has Lambda_{jh}(ell) = tau_sigma * Sigma_h, so its factor
      # in (4.5) is exactly 1 -- but only after cancelling this component's
      # share P_h of the prior term added below as d_j * prior_per_level.
      contrib[n_jh_ell[h, ] == 0] <- -P_h[h]
      contrib[!active_list[[ell_idx]]] <- 0

      spec_term <- spec_term + contrib
    }
  }

  shared_term + constant_term + d_j * prior_per_level + spec_term
}

# Composite (cluster, category) indicator for response dimension t:
# an n x (k * m_t) sparse matrix with a single one per row in column
# (z_i - 1) * m_t + y_{it}. Collapses the k * m_t separate cluster-by-category
# indicators into one design matrix, so all the counts n_{jhts}(ell) for a
# given (t, ell) come from a single crossprod.
build_composite_labels <- function(z, Y, m, k) {
  n <- length(z)
  lapply(seq_along(m), \(t) {
    Matrix::sparseMatrix(
      i = seq_len(n),
      j = (z - 1L) * m[t] + Y[, t],
      x = 1,
      dims = c(n, k * m[t])
    )
  })
}

# log BF_{j2} for categorical kernel
compute_log_bf2_multinomial <- function(
  C,
  psi,
  G_list,
  X_ell_list,
  active_list,
  tau_psi,
  d_j,
  n_jh_ell_list
) {
  # psi: list of length q, psi[[t]] is a k x m_t matrix
  # G_list[[t]]: n x (k * m_t) composite indicator from build_composite_labels
  # n_jh_ell_list[[ell + 1]]: k x p matrix of n_{jh}(ell), shared with BF_{j1}
  k <- ncol(C)
  p <- length(active_list[[1]])
  q <- length(psi)

  # psi entries laid out to match the composite column order (s fastest)
  psi_flat <- lapply(psi, \(P) as.vector(t(P)))
  alpha_flat <- lapply(psi_flat, \(v) tau_psi * v)

  # Shared term: sum_h sum_t sum_s n_{hts} * log(psi_{hts}). The column counts
  # of G_list[[t]] are exactly the n_{hts} in the same order.
  shared_term <- sum(vapply(
    seq_len(q),
    \(t) sum(diff(G_list[[t]]@p) * log(psi_flat[[t]])),
    numeric(1)
  ))

  # Prior per level: sum_h sum_t [sum_s lgamma(tau_psi * psi_{hts})
  #                               - lgamma(tau_psi)]
  prior_per_level <- sum(vapply(
    seq_len(q),
    \(t) sum(lgamma(alpha_flat[[t]])) - k * lgamma(tau_psi),
    numeric(1)
  ))

  # Data term (vectorized over j)
  spec_term <- numeric(p)
  for (ell_idx in seq_along(X_ell_list)) {
    X_ell <- X_ell_list[[ell_idx]]
    n_jh_ell <- n_jh_ell_list[[ell_idx]]

    # lgamma(n_{jh}(ell) + tau_psi) does not depend on t
    contrib <- -q * colSums(lgamma(n_jh_ell + tau_psi))
    for (t in seq_len(q)) {
      # (k * m_t) x p counts n_{jhts}(ell); alpha_flat recycles down columns
      n_jhts_ell <- as.matrix(Matrix::crossprod(G_list[[t]], X_ell))
      contrib <- contrib + colSums(lgamma(n_jhts_ell + alpha_flat[[t]]))
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
  d_j,
  n_jh_ell_list
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
    n_jh_ell <- n_jh_ell_list[[ell_idx]] # k x p

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

# Number of threads used by the "cpp" backend. Defaults to half the physical
# cores, overridable with options(bmfvs.threads = ...).
default_threads <- function() {
  n <- getOption("bmfvs.threads", NULL)
  if (!is.null(n)) {
    return(max(1L, as.integer(n)))
  }
  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores)) {
    return(1L)
  }
  max(1L, as.integer(cores %/% 2L))
}

# General function to compute log Bayes factors
#
# backend = "cpp" evaluates Proposition 2.1 in a single pass over observations
# per predictor, shared by all four kernels, with the lgamma() values read from
# lookup tables indexed by the cell counts and the j loop parallelized.
# backend = "r" is the reference implementation: a direct transcription of
# (4.5), (4.9) and (4.13). The two agree to floating point.
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
  tau_rho = NULL,
  backend = c("cpp", "r"),
  n_threads = default_threads()
) {
  backend <- match.arg(backend)
  S <- nrow(fit$z)
  n <- ncol(fit$z)
  k <- ncol(fit$w)

  qN <- if (is.null(Y_N)) 0L else ncol(Y_N)
  qC <- if (is.null(Y_C)) 0L else ncol(Y_C)
  qP <- if (is.null(Y_P)) 0L else ncol(Y_P)

  # Predictor level encoding (shared across samples and kernels)
  levels <- build_predictor_levels(X, indicators = (backend == "r"))
  d_j <- levels$d_j
  active_list <- levels$active_list
  p <- length(d_j)

  if (backend == "cpp") {
    return(compute_log_bfs_cpp(
      fit, Y_N, Y_C, Y_P, X, d_j, S, n, k, p, qN, qC, qP,
      tau_omega, tau_mu, tau_Sigma, tau_psi, tau_rho, n_threads
    ))
  }

  X_ell_list <- levels$X_ell_list
  n_j_ell_list <- levels$n_j_ell_list
  if (qC > 0) {
    m <- sapply(fit$psi, \(x) dim(x)[3])
  }

  results <- lapply(
    seq_len(S),
    function(s) {
      z_s <- fit$z[s, ]
      C <- Matrix::sparseMatrix(i = seq_len(n), j = z_s, x = 1, dims = c(n, k))
      w_s <- fit$w[s, ]

      # n_{jh}(ell) is needed by every kernel; compute it once per sample
      n_jh_ell_list <- compute_level_counts(C, X_ell_list)

      log_bf1_s <- compute_log_bf1(
        C,
        w_s,
        n_j_ell_list,
        active_list,
        tau_omega,
        n_jh_ell_list
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
            d_j,
            n_jh_ell_list,
            split(seq_len(n), factor(z_s, levels = seq_len(k)))
          )
      }

      if (qC > 0) {
        psi_s <- lapply(fit$psi, \(x) matrix(x[s, , ], nrow = k))
        log_bf2_s <- log_bf2_s +
          compute_log_bf2_multinomial(
            C,
            psi_s,
            build_composite_labels(z_s, Y_C, m, k),
            X_ell_list,
            active_list,
            tau_psi,
            d_j,
            n_jh_ell_list
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
            d_j,
            n_jh_ell_list
          )
      }

      list(log_bf1 = log_bf1_s, log_bf2 = log_bf2_s)
    }
  )

  log_bf1 <- do.call(rbind, lapply(results, `[[`, "log_bf1"))
  log_bf2 <- do.call(rbind, lapply(results, `[[`, "log_bf2"))
  list(log_bf1 = log_bf1, log_bf2 = log_bf2)
}

# Driver for the compiled backend: marshals one posterior sample at a time
# into log_bf_sample_cpp(). Empty blocks are passed as zero-column matrices.
compute_log_bfs_cpp <- function(
  fit, Y_N, Y_C, Y_P, X, d_j, S, n, k, p, qN, qC, qP,
  tau_omega, tau_mu, tau_Sigma, tau_psi, tau_rho, n_threads
) {
  storage.mode(X) <- "integer"
  d_j <- as.integer(d_j)

  empty_num <- matrix(numeric(0), n, 0)
  Y_N_c <- if (qN > 0) Y_N else empty_num
  Y_P_c <- if (qP > 0) Y_P else empty_num
  Y_C_c <- if (qC > 0) {
    matrix(as.integer(Y_C) - 1L, n, qC) # 0-based categories
  } else {
    matrix(integer(0), n, 0)
  }
  storage.mode(Y_N_c) <- "double"
  storage.mode(Y_P_c) <- "double"

  no_mu <- list()
  no_rho <- matrix(numeric(0), k, 0)

  log_bf1 <- matrix(0, S, p)
  log_bf2 <- matrix(0, S, p)

  for (s in seq_len(S)) {
    mu_s <- if (qN > 0) lapply(seq_len(k), \(h) fit$mu[s, h, ]) else no_mu
    Sigma_s <- if (qN > 0) {
      lapply(seq_len(k), \(h) {
        M <- fit$Sigma[s, h, , , drop = FALSE]
        dim(M) <- c(qN, qN)
        M
      })
    } else {
      no_mu
    }
    psi_s <- if (qC > 0) {
      lapply(fit$psi, \(x) matrix(x[s, , ], nrow = k))
    } else {
      no_mu
    }
    rho_s <- if (qP > 0) matrix(fit$rho[s, , ], nrow = k) else no_rho

    res <- log_bf_sample_cpp(
      as.integer(fit$z[s, ]) - 1L,
      X,
      d_j,
      k,
      as.numeric(fit$w[s, ]),
      tau_omega,
      Y_N_c, mu_s, Sigma_s,
      if (is.null(tau_mu)) 0 else tau_mu,
      if (is.null(tau_Sigma)) 0 else tau_Sigma,
      Y_C_c, psi_s,
      if (is.null(tau_psi)) 0 else tau_psi,
      Y_P_c, rho_s,
      if (is.null(tau_rho)) 0 else tau_rho,
      as.integer(n_threads)
    )
    log_bf1[s, ] <- res$log_bf1
    log_bf2[s, ] <- res$log_bf2
  }

  list(log_bf1 = log_bf1, log_bf2 = log_bf2)
}
